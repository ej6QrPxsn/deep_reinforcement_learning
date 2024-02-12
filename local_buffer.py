from typing import NamedTuple
import numpy as np
from config import Config
from data_type import DataType


class Transition(NamedTuple):
  state: np.ndarray
  action: np.uint8
  reward: np.float32
  done: bool


class LocalBuffer:
  def __init__(self, config: Config, data_type: DataType) -> None:
    self._config = config
    self._seq_len = config.context_length
    self._episode_count = 0
    self._index = 0
    self._count = 0

    self._work_transition = np.zeros(self._seq_len, dtype=data_type.work_transition_dtype)
    self.transition = np.zeros(1, dtype=data_type.transition_dtype)

  def get_transition_data(self):
    if self._count < self._seq_len:
      seq_len = self._count
    else:
      seq_len = self._seq_len

    transition = {}
    transition["timestep"] = np.array(self._count)

    transition["state"] = np.expand_dims(self._work_transition["state"][:seq_len], 0)
    transition["action"] = np.expand_dims(self._work_transition["action"][:seq_len], 0)
    transition["rtg"] = np.expand_dims(self._work_transition["reward"][:seq_len], 0)

    return transition

  def get_replay_data(self):
    if self._count < self._seq_len:
      seq_len = self._count
    else:
      seq_len = self._seq_len

    self.transition[:] = 0

    rtg = np.zeros(seq_len, dtype=np.float32)
    # [1 2 3 4 5 6]
    # [1 3 6 10 15 21]
    # 報酬を足し合わせる
    cumsum_rewards = np.cumsum(self._work_transition["reward"][:seq_len])
    # return to go [21 20 18 15 11 6]
    rtg[0] = cumsum_rewards[-1]
    rtg[1:] = rtg[0] - cumsum_rewards[1:seq_len]

    self.transition["state"][0][:seq_len] = self._work_transition["state"][:seq_len]
    self.transition["action"][0][:seq_len] = self._work_transition["action"][:seq_len]
    self.transition["rtg"][0][:seq_len] = rtg
    self.transition["timestep"][0] = self._count

    return self.transition.tobytes()

  def add_and_get_replay_data(self, tansition: Transition):
    self._work_transition["state"][self._index] = tansition.state
    self._work_transition["action"][self._index] = tansition.action
    self._work_transition["reward"][self._index] = tansition.reward

    self._count += 1

    # エピソード終了
    if tansition.done or self._count >= self._config.max_timestep:
      self._index = 0

      data = self.get_replay_data()
      self._count = 0
      return data
    else:
      # データ追加用に末尾を空ける
      self._work_transition[:-1] = self._work_transition[1:]

      # エピソード開始後、バッファが埋まっていない
      if self._index < self._seq_len - 1:
        self._index += 1
      else:
        data = self.get_replay_data()
        return data

  def add_and_get_transition_data(self, tansition: Transition):
    self._work_transition["state"][self._index] = tansition.state
    self._work_transition["action"][self._index] = tansition.action
    self._work_transition["reward"][self._index] = tansition.reward

    self._count += 1

    # エピソード終了
    if tansition.done or self._count >= self._config.max_timestep:
      self._index = 0

      data = self.get_transition_data()
      self._count = 0
      return data
    else:
      # データ追加用に末尾を空ける
      self._work_transition[:-1] = self._work_transition[1:]

      # エピソード開始後、バッファが埋まっていない
      if self._index < self._seq_len - 1:
        self._index += 1

      data = self.get_transition_data()
      return data
