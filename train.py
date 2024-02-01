import os
from pathlib import Path
import pickle
import time
import gymnasium
import zstandard as zstd
import numpy as np
import torch
from config import Config
from data_type import DataType
from data_writer import DataWriter, info_from_json
from env import AtariEnv
from local_buffer import LocalBuffer, Transition
import multiprocessing as mp
from model import DecisionTransformer, Input
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import webdataset as wds
from torch.utils.data import DataLoader

from replay_buffer import ReplayBuffer


def env_loop(id, config: Config):
  rng = np.random.default_rng()
  ctx = zstd.ZstdCompressor()

  env = AtariEnv(config.env_name)

  env_output = env.reset()
  state = env_output.next_state

  time.sleep(10)

  if id == 0:
    pber = tqdm(range(config.n_steps))
  else:
    pber = range(config.n_steps)

  writer = DataWriter(id, config)

  for i in pber:
    action = rng.integers(env.action_space)
    env_output = env.step(action)

    data = ctx.compress(
        pickle.dumps(
          Transition(
            state,
            action,
            env_output.reward,
            env_output.done
          )
        )
      )

    writer.write_train_data(data)

    state[:] = env_output.next_state

  writer.write_end()


def create_dataset(config: Config):
  if not os.path.exists(config.train_data_dir):
    os.makedirs(config.train_data_dir)

  if not os.path.exists(config.validate_data_dir):
    os.makedirs(config.validate_data_dir)

  processes = []

  # 訓練データ取得
  for i in range(config.n_envs):
    p = mp.Process(target=env_loop, args=(i, config))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()


def load_loop(id, data_queue, data_type, config):
  shards_list = [
      str(path) for path in Path(config.train_data_dir).glob(f"{id}_*.tar")
  ]

  dataset = wds.WebDataset(shards_list)
  dataset = dataset.to_tuple("bytes")
  dataloader = DataLoader(dataset)
  total = info_from_json(id, config.train_data_dir)

  local_buffer = LocalBuffer(config, data_type)

  cctx = zstd.ZstdCompressor()
  dctx = zstd.ZstdDecompressor()

  data_it = iter(dataloader)

  if id == 0:
    pber = tqdm(range(total))
  else:
    pber = range(total)

  for _ in pber:
    byte_data = next(data_it)[0][0]
    # ファイルから解凍して読み込み
    a = dctx.decompress(byte_data)
    data = pickle.loads(a)

    seq_data = local_buffer.add(data)
    if seq_data:
      # 圧縮してキューに追加
      data_queue.put(cctx.compress(seq_data))

  dataset.close()


def train_loop(config: Config):
  device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_type)
  summary_writer = SummaryWriter("logs")

  data_type = DataType(config)

  net = DecisionTransformer(config, device).to(device)
  net.train()

  opt = torch.optim.Adam(
      params=net.parameters(),
      lr=config.adam_lr,
      betas=config.adam_beta,
  )
  scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

  total_steps = 0
  sample_queue = mp.Queue()
  load_queue = mp.Queue()

  replay = ReplayBuffer(config, data_type, sample_queue)

  processes = []

  p = mp.Process(target=replay.replay_loop, args=(load_queue, ))
  p.start()
  processes.append(p)

  for i in range(config.n_loads):
    p = mp.Process(target=load_loop, args=(i, load_queue, data_type, config))
    p.start()
    processes.append(p)

  while True:
    data = sample_queue.get()
    if data is None:
      break

    input = Input(
      rtg=torch.from_numpy(data["rtg"].astype(np.float32)).unsqueeze(-1).clone().to(device),
      state=torch.from_numpy(data["state"].astype(np.float32) / 255.).clone().to(device),
      action=torch.from_numpy(data["action"].astype(np.float32)).unsqueeze(-1).clone().to(device),
      timestep=torch.from_numpy(data["timestep"]).unsqueeze(-1).to(device),
    )

    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=config.use_amp):
      logits = net(input)
      loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), data.action.to(torch.int64).reshape(-1))
      loss = loss.mean()

    scaler.scale(loss).backward()

    # Unscales the gradients of optimizer's assigned parameters in-place
    scaler.unscale_(opt)

    # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
    # You may use the same value for max_norm here as you would without gradient scaling.
    torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_norm_clip)

    scaler.step(opt)
    scaler.update()
    opt.zero_grad()  # set_to_none=True here can modestly improve performance

    total_steps += 1
    summary_writer.add_scalar("train/loss", loss, total_steps)

    if total_steps % 100 == 0:

      checkpoint = {"model": net.cpu().state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict()}
      torch.save(checkpoint, config.checkpoint_path)
      net.to(device)


def validate_loop(config: Config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  summary_writer = SummaryWriter("logs")

  env = AtariEnv(config.env_name)
  net = DecisionTransformer(config, device).to(device)
  net.eval()

  if os.path.exists(config.checkpoint_path):
    checkpoint = torch.load(config.checkpoint_path)
    net.load_state_dict(checkpoint["model"])

  target_return = 90
  R, s, a, t, done = target_return, env.reset().next_state, 0, 1, False

  total_steps = 0
  tensor_R = torch.empty(1, 1, 1).to(device)
  tensor_s = torch.empty(1, 1, *config.state_shape).to(device)
  tensor_a = torch.empty(1, 1, 1).to(device)
  tensor_t = torch.empty(1, 1, 1).to(device)

  episode_count = 0

  while True:
    episode_reward = 0
    t = 1
    episode_count += 1
    while not done:
      tensor_R[:] = R
      tensor_s[:] = torch.from_numpy(s)
      tensor_a[:] = a
      tensor_t[:] = t
      input = Input(
        rtg=tensor_R,
        state=tensor_s,
        action=tensor_a,
        timestep=tensor_t,
      )

      with torch.no_grad():
        preds = net(input)
      probs = F.softmax(preds, dim=2)
      action = probs.argmax(2).item()
      env_output = env.step(action)

      R = R - env_output.reward
      s = env_output.next_state
      a = action
      t += 1
      done = env_output.done

      episode_reward += env_output.reward
      total_steps += 1

    if episode_count % 5 == 0:
      summary_writer.add_scalar("validate/reward", episode_reward, total_steps)


def set_action_space(config):
  env = gymnasium.make(config.env_name)
  config.action_size = env.action_space.n


if __name__ == "__main__":
  mp.set_start_method("spawn")  # set start method to "spawn" BEFORE instantiating the queue and the event

  config = Config()
  set_action_space(config)

  # create_dataset(config)
  train_loop(config)

  p = mp.Process(target=validate_loop, args=(config, ))
  p.start()
  p.join()
