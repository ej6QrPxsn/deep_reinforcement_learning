import os
import pickle
import time
import gymnasium
import zstandard as zstd
import numpy as np
import torch
from config import Config
from data_loader import SingleDataLoader
from data_type import DataType
from data_writer import DataWriter
from env import AtariEnv, BatchedEnv
from local_buffer import Transition
import multiprocessing as mp
from model import DecisionTransformer, Input
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer


def env_loop(ids, config: Config):
  rng = np.random.default_rng()
  ctx = zstd.ZstdCompressor()

  batched_env = BatchedEnv(config)

  env_output = batched_env.reset()
  states = env_output.next_state.copy()

  time.sleep(10)

  writers = []
  for id in ids:
    writers.append(DataWriter(id, config))

  if 0 in ids:
    pber = tqdm(range(config.n_steps), leave=False)
  else:
    pber = range(config.n_steps)

  batches = np.arange(config.n_env_batches)
  indexes = np.zeros(config.n_env_batches, dtype=np.int32)
  max_timestep = config.max_timestep
  actions = rng.integers(0, config.action_size, (config.n_env_batches, max_timestep))

  for _ in pber:
    current_actions = actions[batches, indexes]
    env_output = batched_env.step(current_actions)

    indexes += 1

    for i in range(config.n_env_batches):
      bytes = states[i].tobytes()
      data = pickle.dumps(
        Transition(
          ctx.compress(states[i].tobytes()),
          current_actions[i],
          env_output.reward[i],
          env_output.done[i]
        )
      )
      writers[i].write_train_data(data)

    done_ids = np.where(env_output.done)[0]
    if len(done_ids) > 0:
      indexes[done_ids] = 0
      actions[done_ids] = rng.integers(0, config.action_size, (len(done_ids), config.max_timestep))

    states[:] = env_output.next_state

  for writer in writers:
    writer.write_end()


def create_dataset(config: Config, ids):
  if not os.path.exists(config.train_data_dir):
    os.makedirs(config.train_data_dir)

  if not os.path.exists(config.validate_data_dir):
    os.makedirs(config.validate_data_dir)

  processes = []

  # 訓練データ取得
  for i in range(config.n_envs):
    p = mp.Process(target=env_loop, args=(ids[i], config))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()


def load_loop(ids, data_queue, data_type, config):
  data_loaders = []
  load_end_flags = np.zeros(len(ids), dtype=bool)
  for id in ids:
    data_loaders.append(SingleDataLoader(id, config, data_type))

  while True:
    for i, data_loader in enumerate(data_loaders):
      if not load_end_flags[i]:
        ret = data_loader.load(data_queue)
        # ローダーの読み込み終了
        if ret is None:
          load_end_flags[i] = True

    end_list = np.where(load_end_flags)[0]
    if len(end_list) == len(data_loaders):
      break


def get_input(data, device):
  return Input(
    rtg=torch.from_numpy((data["rtg"].astype(np.float32)).copy()).unsqueeze(-1).to(device),
    state=torch.from_numpy((data["state"].astype(np.float32)).copy()).to(device),
    action=torch.from_numpy((data["action"].astype(np.float32)).copy()).unsqueeze(-1).to(device),
    timestep=torch.from_numpy(data["timestep"].astype(np.int64).copy()).reshape(-1, 1, 1).to(device),
  )


def run_train_epoch(config: Config, weight_lock, ids, total_steps):
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

  if os.path.exists(config.checkpoint_path):
    checkpoint = torch.load(config.checkpoint_path)
    net.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

  sample_queue = mp.Queue()
  load_queue = mp.SimpleQueue()

  replay = ReplayBuffer(config, data_type)

  processes = []

  p = mp.Process(target=replay.replay_loop, args=(load_queue, sample_queue))
  p.start()
  processes.append(p)

  for i in range(config.n_loads):
    p = mp.Process(target=load_loop, args=(ids[i], load_queue, data_type, config))
    p.start()
    processes.append(p)

  criteria = torch.nn.CrossEntropyLoss(reduction="none")

  data = sample_queue.get()
  while data is not None:

    input = get_input(data, device)

    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=config.use_amp):
      logits = net(input)
      target = input.action[:, -1, 0].to(torch.long)

      loss = criteria(logits, target)
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

    if total_steps % 10 == 0:

      checkpoint = {"model": net.cpu().state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict()}
      with weight_lock:
        torch.save(checkpoint, config.checkpoint_path)
      net.to(device)

    data = sample_queue.get()

  for p in processes:
    p.join()

  return total_steps


def validate_loop(config: Config, weight_lock):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  summary_writer = SummaryWriter("logs")

  env = AtariEnv(config.env_name, config.max_timestep)
  net = DecisionTransformer(config, device).to(device)
  net.eval()

  # 訓練が始まるまでループ
  while True:
    if os.path.exists(config.checkpoint_path):
      with weight_lock:
        checkpoint = torch.load(config.checkpoint_path)
      net.load_state_dict(checkpoint["model"])
      break
    else:
      time.sleep(10)

  target_return = 90

  total_steps = 0
  tensor_R = torch.zeros(1, 1, 1).to(device)
  tensor_s = torch.zeros(1, 1, *config.state_shape).to(device)
  tensor_a = torch.zeros(1, 1, 1).to(device)
  tensor_t = torch.zeros(1, 1, 1).to(torch.int64).to(device)

  episode_count = 0

  while True:
    R, s, a, t, done = target_return, env.reset().next_state, 0, 1, False

    episode_reward = 0
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
        logits = net(input)
      probs = F.softmax(logits, dim=-1)
      action = torch.multinomial(probs, num_samples=1).item()
      env_output = env.step(action)

      R = R - env_output.reward
      s = env_output.next_state
      a = action
      t += 1
      done = env_output.done

      episode_reward += env_output.reward
      total_steps += 1

      if total_steps % 10 == 0:
        if os.path.exists(config.checkpoint_path):
          with weight_lock:
            checkpoint = torch.load(config.checkpoint_path)
          net.load_state_dict(checkpoint["model"])

    summary_writer.add_scalar("validate/reward", episode_reward, total_steps)


def set_action_space(config):
  env = gymnasium.make(config.env_name)
  config.action_size = env.action_space.n


if __name__ == "__main__":
  mp.set_start_method("spawn")  # set start method to "spawn" BEFORE instantiating the queue and the event

  config = Config()
  set_action_space(config)

  ids = np.arange(config.n_envs * config.n_env_batches)
  ids = ids.reshape(config.n_envs, -1)

  if not os.path.exists(config.train_data_dir):
    create_dataset(config, ids)

  weight_lock = mp.Lock()

  p = mp.Process(target=validate_loop, args=(config, weight_lock))
  p.start()

  total_steps = 0
  for i in range(config.max_epochs):
    total_steps = run_train_epoch(config, weight_lock, ids, total_steps)

  p.join()
