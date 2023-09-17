import os
from typing import NamedTuple

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

from env import EnvOutput


class ActorOutput(NamedTuple):
  next_state: np.ndarray
  reward: np.ndarray
  done: np.ndarray
  meta_index: np.ndarray


class SharedData():
  def __init__(self, name, shape, dtype):
    self._name = name
    self._shape = shape
    self._dtype = dtype
    self._memory = None
    self.data = None

  def __del__(self):
    self._memory.close()
    self._memory.unlink()

  # 共有メモリを新規作成
  def create_shared_memory(self):
    if (os.path.isfile(f"/dev/shm/{self._name}")):
      os.remove(f"/dev/shm/{self._name}")

    self._memory = shared_memory.SharedMemory(create=True, size=self._dtype.itemsize * np.prod(self._shape),
                                              name=self._name)

  # 作成済み共有メモリを取得
  def get_shared_memory(self):
    self._memory = shared_memory.SharedMemory(name=self._name)
    # ndarrayオブジェクトを作成
    self.data = np.ndarray(
        shape=self._shape, dtype=self._dtype, buffer=self._memory.buf)


# アクターとの情報共有
class SharedActorData():

  def __init__(self, ids, shared_data):

    self._actor_event = mp.Event()
    self._action_event = mp.Event()
    self.shared = shared_data
    self._ids = ids

  def put_actor_data(self, env_output: EnvOutput, meta_index):
    self.shared.data["next_state"][self._ids] = env_output.next_state
    self.shared.data["reward"][self._ids] = env_output.reward
    self.shared.data["done"][self._ids] = env_output.done
    self.shared.data["meta_index"][self._ids] = meta_index
    self._actor_event.set()

  def get_actor_data(self):
    self._actor_event.wait()
    actor_output = ActorOutput(
      next_state=self.shared.data["next_state"][self._ids],
      reward=self.shared.data["reward"][self._ids],
      done=self.shared.data["done"][self._ids],
      meta_index=self.shared.data["meta_index"][self._ids],
    )

    self._actor_event.clear()

    return self._ids, actor_output

  def put_action(self, action):
    self.shared.data["action"][self._ids] = action
    self._action_event.set()

  def get_action(self):
    self._action_event.wait()
    action = self.shared.data["action"][self._ids]
    self._action_event.clear()

    return action
