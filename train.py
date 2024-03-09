import os
from agent import Agent
import gymnasium
import numpy as np
import torch
from config import Config
from data_loader import SingleDataLoader
from data_type import DataType
import multiprocessing as mp
from model import Input
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from target_manager import TargetManager


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


def train(config: Config):
  agent = Agent(config)
  device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_type)
  data_type = DataType(config)
  summary_writer = SummaryWriter("logs")

  def run_train_epoch(ids):
    steps = 0
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

    data = sample_queue.get()

    while data is not None:

      input = get_input(data, device)

      loss = agent.train(input)

      steps += 1
      summary_writer.add_scalar("train/loss", loss, steps)

      if steps % 2000 == 0:
        reward = agent.eval()
        summary_writer.add_scalar("eval/reward", reward, steps)

      data = sample_queue.get()

    for p in processes:
      p.join()

    return steps

  for i in range(config.max_epochs):
    run_train_epoch(ids)


def set_action_space(config):
  env = gymnasium.make(config.env_name)
  config.action_size = env.action_space.n


def create_dataset(config):
  manager = TargetManager(config)

  processes = []
  for i in range(config.n_writers):
    data_writer = manager.writer(i, config)
    p = mp.Process(target=data_writer.write)
    p.start()
    processes.append(p)

  for p in processes:
    p.join()


if __name__ == "__main__":
  mp.set_start_method("spawn")  # set start method to "spawn" BEFORE instantiating the queue and the event

  config = Config()
  set_action_space(config)

  ids = np.arange(config.n_envs)
  ids = ids.reshape(config.n_envs, -1)

  if not os.path.exists(config.train_data_dir):
    os.makedirs(config.train_data_dir)

  # create_dataset(config)

  train(config)
