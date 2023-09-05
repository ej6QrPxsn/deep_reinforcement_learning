import os

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

from env import EnvOutput


class SharedData():
  def __init__(self, name, shape, dtype):
    self.name = name
    self.shape = shape
    self.dtype = dtype
    self.memory = None
    self.data = None

  def __del__(self):
    self.memory.close()
    self.memory.unlink()

  # 共有メモリを新規作成
  def create_shared_memory(self):
    if (os.path.isfile(f"/dev/shm/{self.name}")):
      os.remove(f"/dev/shm/{self.name}")

    self.memory = shared_memory.SharedMemory(create=True, size=self.dtype.itemsize * np.prod(self.shape),
                                             name=self.name)

  # 作成済み共有メモリを取得
  def get_shared_memory(self):
    self.memory = shared_memory.SharedMemory(name=self.name)
    # ndarrayオブジェクトを作成
    self.data = np.ndarray(
        shape=self.shape, dtype=self.dtype, buffer=self.memory.buf)


# アクターとの情報共有
class SharedEnvData():

  def __init__(self, ids, shared_data):

    self.env_event = mp.Event()
    self.action_event = mp.Event()
    self.shared = shared_data
    self.ids = ids

  def put_env_data(self, env_output: EnvOutput, policy_index):
    self.shared.data["next_state"][self.ids] = env_output.next_state
    self.shared.data["reward"][self.ids] = env_output.reward
    self.shared.data["done"][self.ids] = env_output.done
    self.shared.data["policy_index"][self.ids] = policy_index
    self.env_event.set()

  def get_env_data(self):
    self.env_event.wait()
    env_output = EnvOutput(
      next_state=self.shared.data["next_state"][self.ids],
      reward=self.shared.data["reward"][self.ids],
      done=self.shared.data["done"][self.ids],
    )
    policy_index = self.shared.data["policy_index"][self.ids]

    self.env_event.clear()

    return self.ids, env_output, policy_index

  def put_action(self, action):
    self.shared.data["action"][self.ids] = action
    self.action_event.set()

  def get_action(self):
    self.action_event.wait()
    action = self.shared.data["action"][self.ids]
    self.action_event.clear()

    return action
