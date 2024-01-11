import datetime as dt
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
import torch
from local_buffer import LocalBuffer, Transition
import multiprocessing as mp
from model import DecisionTransformer, Input
from tqdm import tqdm
import torch.nn.functional as F


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
  while data:
    now = dt.datetime.now()
    time = now.strftime("%Y%m%d-%H%M%S-%f")
    key_str = time

    # 7割の確率で訓練データ
    if random.random() < 0.7:
      train_writer.write({
          "__key__": key_str,
          "bytes": ctx.compress(data),
      })
      train_count += 1
    # 3割の確率でテストデータ
    else:
      validate_writer.write({
          "__key__": key_str,
          "bytes": ctx.compress(data),
      })
      validate_count += 1

    # 空データで終了
    data = data_queue.get()

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


def get_dataset(shard_dir, config, device):

  def info_from_json(shard_dir):
    with open(Path(shard_dir) / 'dataset-size.json', 'r') as f:
      info_dic = json.load(f)
    return info_dic['dataset size']

  shards_list = [
      str(path) for path in Path(shard_dir).glob('*.tar')
  ]

  dctx = zstd.ZstdDecompressor()
  data_type = DataType(config)

  def ready_data(bytes):
    data = np.frombuffer(dctx.decompress(bytes, max_output_size=data_type.transition_dtype.itemsize),
                         dtype=data_type.transition_dtype)[0].copy()
    timestep = np.arange(data["timestep"] - config.context_length, data["timestep"])

    return Input(
      rtg=torch.from_numpy(data["rtg"].astype(np.float32)).unsqueeze(-1).clone().to(device),
      state=torch.from_numpy(data["state"].astype(np.float32)).clone().to(device),
      action=torch.from_numpy(data["action"].astype(np.float32)).unsqueeze(-1).clone().to(device),
      timestep=torch.from_numpy(timestep).unsqueeze(-1).to(device),
    )

  dataset = wds.WebDataset(shards_list)
  dataset = dataset.to_tuple("bytes")
  dataset = dataset.map_tuple(ready_data)
  dataset_size = info_from_json(shard_dir)
  dataset = dataset.with_length(dataset_size)

  return dataset


def train_loop(config: Config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  dataset = get_dataset(config.train_data_dir, config, device)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=4)

  decision_transformer = DecisionTransformer(config)
  optimizer = torch.optim.Adam(
      params=decision_transformer.parameters(),
      lr=config.adam_lr,
      betas=config.adam_beta,
  )

  # training loop
  for _ in range(config.max_epochs):
    for ret in dataloader:
      data = ret[0]
      logits = decision_transformer(data)

      # 期待値をone-hotにする
      targets = F.one_hot(data.action.to(torch.int64), num_classes=config.action_size)
      targets = targets.squeeze(2).to(torch.float32)

      loss = F.cross_entropy(logits, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


def set_action_space(config):
  env = gymnasium.make(config.env_name)
  config.action_size = env.action_space.n


if __name__ == "__main__":
  config = Config()
  set_action_space(config)
  create_dataset(config)
  train_loop(config)
