import datetime as dt
import json
import os
import random

import numpy as np
from config import Config
import webdataset as wds
from data_type import DataType
from env import AtariEnv, EnvOutput

from local_buffer import LocalBuffer, Transition
import multiprocessing as mp


def write_train_data(train_writer, validate_writer, data_queue):
  train_count = 0
  validate_count = 0

  data: Transition = data_queue.get()
  while data:
    now = dt.datetime.now()
    time = now.strftime("%Y%m%d-%H%M%S.%f")
    key_str = time

    # 7割の確率で訓練データ
    if random.random() < 0.7:
      train_writer.write({
          "__key__": key_str,
          "pyd": data,
          "cls": data.action[-1],
      })
      train_count += 1
    # 3割の確率でテストデータ
    else:
      validate_writer.write({
          "__key__": key_str,
          "pyd": data,
          "cls": data.action[-1],
      })
      validate_count += 1

    # 空データで終了
    data = data_queue.get()

  last_write_data(train_count, validate_count)
  print("write_train_data end")


def last_write_data(train_count, validate_count):
  dataset_size_filename = f"{config.data_dir}/train_dataset-size.json"
  with open(dataset_size_filename, 'w') as fp:
    json.dump({
        "dataset size": train_count,
    }, fp)

  dataset_size_filename = f"{config.data_dir}/validate_dataset-size.json"
  with open(dataset_size_filename, 'w') as fp:
    json.dump({
        "dataset size": validate_count,
    }, fp)


def env_loop(config: Config, data_type, data_queue):
  rng = np.random.default_rng()
  local_buffer = LocalBuffer(config, data_type, data_queue)

  env = AtariEnv(config.env_name)

  state = env.reset()

  env_output = EnvOutput(
    next_state=state,
    reward=0,
    done=False,
  )

  for _ in range(config.n_steps):
    state[:] = env_output.next_state
    action = rng.integers(env.action_space)
    env_output = env.step(action)
    local_buffer.add(state, env_output.reward, action, env_output.done)


def create_dataset(config: Config):
  data_queue = mp.Queue(maxsize=config.data_queue_max_size)
  data_type = DataType(config)

  if not os.path.exists(config.data_dir):
    os.makedirs(config.data_dir)

  train_writer = wds.ShardWriter(
      pattern=config.train_filename,
      maxsize=config.shard_size,
  )
  validate_writer = wds.ShardWriter(
      pattern=config.validate_filename,
      maxsize=config.shard_size,
  )

  processes = []

  # 訓練データ書き込み
  p = mp.Process(target=write_train_data, args=(train_writer, validate_writer, data_queue))
  p.start()
  processes.append(p)

  # 訓練データ取得
  for i in range(config.n_envs):
    p = mp.Process(target=env_loop, args=(config, data_type, data_queue))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()


# def train_loop():
#   # training loop
#   for (R, s, a, t) in dataloader:  # dims : ( batch_size , K, dim )
#     a_preds = DecisionTransformer(R, s, a, t)
#     loss = torch.mean((a_preds - a)**2)  # L2 loss for continuous actions
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


if __name__ == "__main__":
  config = Config()
  create_dataset(config)
