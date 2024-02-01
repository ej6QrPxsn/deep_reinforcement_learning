import datetime as dt
from functools import partial
import json
import os
from pathlib import Path
import random
import gymnasium
import zstandard as zstd
import numpy as np
import torch
from config import Config
import webdataset as wds
from data_type import DataType
from env import AtariEnv
from local_buffer import LocalBuffer, Transition
import multiprocessing as mp
from model import DecisionTransformer, Input
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def write_train_data(data_queue, config):
  train_writer = wds.ShardWriter(
      pattern=f"file:{config.train_data_dir}/{config.train_filename}",
      maxsize=config.shard_size,
      verbose=0,
  )
  validate_writer = wds.ShardWriter(
      pattern=f"file:{config.validate_data_dir}/{config.validate_filename}",
      maxsize=config.shard_size,
      verbose=0,
  )

  train_count = 0
  validate_count = 0
  ctx = zstd.ZstdCompressor()

  data: Transition = data_queue.get()
  end_count = 0

  while True:
    now = dt.datetime.now()
    time = now.strftime("%Y%m%d-%H%M%S-%f")
    key_str = time

    if random.random() < config.train_date_ratio:
      train_writer.write({
          "__key__": key_str,
          "bytes": ctx.compress(data),
      })
      train_count += 1
    else:
      validate_writer.write({
          "__key__": key_str,
          "bytes": ctx.compress(data),
      })
      validate_count += 1

    # 空データで終了
    data = data_queue.get()
    if not data:
      end_count += 1
      if end_count == config.n_envs:
        break

  last_write_data(train_count, validate_count, config)
  print("write_train_data end")


def last_write_data(train_count, validate_count, config):
  dataset_size_filename = f"{config.train_data_dir}/dataset-size.json"
  with open(dataset_size_filename, 'w') as fp:
    json.dump({
        "dataset size": train_count,
    }, fp)

  dataset_size_filename = f"{config.validate_data_dir}/dataset-size.json"
  with open(dataset_size_filename, 'w') as fp:
    json.dump({
        "dataset size": validate_count,
    }, fp)


def env_loop(config: Config, data_type, data_queue):
  rng = np.random.default_rng()
  local_buffer = LocalBuffer(config, data_type, data_queue)

  env = AtariEnv(config.env_name)

  env_output = env.reset()
  state = env_output.next_state

  for i in tqdm(range(config.n_steps), leave=False):
    action = rng.integers(env.action_space)
    env_output = env.step(action)
    local_buffer.add(state, env_output.reward, action, env_output.done)
    state[:] = env_output.next_state

  data_queue.put(None)


def create_dataset(config: Config):
  data_queue = mp.Queue(maxsize=config.data_queue_max_size)
  data_type = DataType(config)

  if not os.path.exists(config.train_data_dir):
    os.makedirs(config.train_data_dir)

  if not os.path.exists(config.validate_data_dir):
    os.makedirs(config.validate_data_dir)

  processes = []

  # 訓練データ書き込み
  p = mp.Process(target=write_train_data, args=(data_queue, config))
  p.start()
  processes.append(p)

  # 訓練データ取得
  for i in range(config.n_envs):
    p = mp.Process(target=env_loop, args=(config, data_type, data_queue))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()


def ready_data(context_length, data_type, device, bytes):
  dctx = zstd.ZstdDecompressor()
  data = np.frombuffer(dctx.decompress(bytes, max_output_size=data_type.transition_dtype.itemsize),
                       dtype=data_type.transition_dtype)[0].copy()
  timestep = np.arange(data["timestep"] - context_length, data["timestep"])

  return Input(
    rtg=torch.from_numpy(data["rtg"].astype(np.float32)).unsqueeze(-1).clone().to(device),
    state=torch.from_numpy(data["state"].astype(np.float32) / 255.).clone().to(device),
    action=torch.from_numpy(data["action"].astype(np.float32)).unsqueeze(-1).clone().to(device),
    timestep=torch.from_numpy(timestep).unsqueeze(-1).to(device),
  )


def get_dataset(shard_dir, data_type, device):

  shards_list = [
      str(path) for path in Path(shard_dir).glob('*.tar')
  ]

  data_func = partial(
      ready_data,
      config.context_length, data_type, device)

  dataset = wds.WebDataset(shards_list)
  dataset = dataset.to_tuple("bytes")
  dataset = dataset.map_tuple(data_func)

  return dataset


def info_from_json(shard_dir):
  with open(Path(shard_dir) / 'dataset-size.json', 'r') as f:
    info_dic = json.load(f)
  return int(info_dic['dataset size'])


def train_loop(config: Config):
  device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_type)
  summary_writer = SummaryWriter("logs")

  data_type = DataType(config)

  dataset = get_dataset(config.train_data_dir, data_type, device)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=4)

  net = DecisionTransformer(config, device).to(device)
  net.train()

  opt = torch.optim.Adam(
      params=net.parameters(),
      lr=config.adam_lr,
      betas=config.adam_beta,
  )
  scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

  total_steps = 0

  dataset_size = info_from_json(config.train_data_dir)
  total = dataset_size // config.batch_size
  total = 2 * 500000 // config.batch_size

  epoch_pbar = tqdm(range(config.max_epochs))
  for _ in epoch_pbar:
    pbar = tqdm(dataloader, total=total)
    for ret in pbar:
      data = ret[0]

      with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=config.use_amp):
        logits = net(data)
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

  for i in range(config.n_val_episode):
    episode_reward = 0
    t = 1
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
  validate_loop(config)
