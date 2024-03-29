

import numpy as np
import torch
import multiprocessing as mp
from actor import actor_loop, tester_loop
from config import Config
from data_type import DataType

from env import AtariEnv
from learner import eval_loop, inference_loop, train_loop
from models import Agent57Network, EmbeddingNetwork, RNDPredictionNetwork
from replay_buffer import replay_loop

import os

from shared_data import SharedActorData, SharedData

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__":
  ################################################################################
  mp.set_start_method("spawn")  # set start method to "spawn" BEFORE instantiating the queue and the event
  ################################################################################

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  transition_queue = mp.Queue()
  sample_queue = mp.Queue()
  priority_queue = mp.Queue()

  config = Config()
  env = AtariEnv(config.env_name)
  config.init(env.action_space, env.reset().shape)

  if not os.path.exists(config.train_data_dir):
    os.makedirs(config.train_data_dir)

  if not os.path.exists(config.validate_data_dir):
    os.makedirs(config.validate_data_dir)

  data_type = DataType(config)

  processes = []
  p = mp.Process(target=replay_loop, args=(transition_queue, sample_queue, priority_queue, config))
  p.start()
  processes.append(p)

  infer_net = Agent57Network(device, config)
  infer_net.share_memory()

  predict_net = RNDPredictionNetwork(device, config)
  embedding_net = EmbeddingNetwork(device, config)

  env_ids = np.arange(config.num_train_envs).reshape(config.num_actors, config.num_env_batches)
  log_ids = np.linspace(0, config.num_train_envs - 1, num=config.num_train_log, dtype=int)

  shared_data = SharedData(
    config.shared_env_name,
    (config.num_train_envs + config.num_eval_envs, ),
    data_type.env_dtype
  )
  shared_data.create_shared_memory()

  # アクターデータ共有
  shared_actor_datas = np.empty(config.num_actors + config.num_eval_envs, dtype=object)
  for i in range(config.num_actors):
    shared_actor_datas[i] = SharedActorData(env_ids[i], shared_data)
    p = mp.Process(target=actor_loop, args=(env_ids[i], log_ids, shared_actor_datas[i], config))
    p.start()
    processes.append(p)
  shared_actor_datas[-1] = SharedActorData([config.num_train_envs], shared_data)

  # 推論プロセスごとのアクター
  inference_actor_indexes = np.split(np.arange(config.num_actors), config.num_inferences)

  for i in range(config.num_inferences):
    actor_indexes = inference_actor_indexes[i]
    p = mp.Process(target=inference_loop, args=(actor_indexes, infer_net, predict_net, embedding_net, transition_queue, shared_actor_datas[actor_indexes], device, config))
    p.start()
    processes.append(p)

  eval_env_queue = mp.Queue()
  eval_action_queue = mp.SimpleQueue()

  p = mp.Process(target=tester_loop, args=(shared_actor_datas[-1], config))
  p.start()
  processes.append(p)

  p = mp.Process(target=eval_loop, args=(infer_net, predict_net, embedding_net, config, shared_actor_datas[-1], device))
  p.start()
  processes.append(p)

  p = mp.Process(target=train_loop, args=(0, infer_net, predict_net, embedding_net, sample_queue, priority_queue, device, config))
  p.start()
  processes.append(p)

  for p in processes:
    p.join()
