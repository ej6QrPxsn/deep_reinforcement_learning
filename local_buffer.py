from typing import NamedTuple
import numpy as np
from config import Config
from data_type import DataType


class Transition(NamedTuple):
  state: np.ndarray
  rtg: np.ndarray
  action: np.ndarray
  timestep: int


class LocalBuffer:
  def __init__(self, config: Config, data_type: DataType, data_queue) -> None:
    self._config = config
    self._seq_len = config.context_length
    self._data_queue = data_queue
    self._episode_count = 0
    self._index = 0
    self._count = 0

    self._work_transition = np.zeros(self._seq_len, dtype=data_type.work_transition_dtype)
    self.transition = np.zeros(1, dtype=data_type.transition_dtype)

  def set_queue(self):
    # 報酬を逆順に
    reverse_rewards = self._work_transition["reward"][::-1]
    # 足し合わせて逆順にすると、最後までの報酬合計が得られる
    total_rewards = np.cumsum(reverse_rewards)[::-1]

    self.transition["state"][0][:] = self._work_transition["state"]
    self.transition["action"][0][:] = self._work_transition["action"]
    self.transition["rtg"][0][:] = total_rewards
    self.transition["timestep"][0] = self._episode_count

    self._data_queue.put(self.transition.tobytes())

  def add(self, state, reward, action, done):
    self._work_transition["state"][self._index] = state
    self._work_transition["action"][self._index] = action
    self._work_transition["reward"][self._index] = reward

    # エピソード終了
    if done:
      self.set_queue()

      self._index = 0
      self._count = 0
      self._episode_count = 0
    else:
      # エピソード開始後、バッファが埋まっていない
      if self._index < self._seq_len - 1:
        self._index += 1

      self._count += 1
      self._episode_count += 1

      # 蓄積長さ
      if self._count == self._seq_len - 1:
        self.set_queue()
        self._count = 0

      # データ追加用に末尾を空ける
      self._work_transition[:-1] = self._work_transition[1:]
