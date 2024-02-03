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

  def get_data(self):
    # 報酬を逆順に
    reverse_rewards = self._work_transition["reward"][::-1]
    # 足し合わせて逆順にすると、最後までの報酬合計が得られる
    total_rewards = np.cumsum(reverse_rewards)[::-1]

    self.transition["state"][0][:] = self._work_transition["state"]
    self.transition["action"][0][:] = self._work_transition["action"]
    self.transition["rtg"][0][:] = total_rewards
    if self._count > self._config.context_length:
      self.transition["timestep"][0] = self._count - self._seq_len
    else:
      self.transition["timestep"][0] = 0

    return self.transition.tobytes()

  def add(self, tansition: Transition):
    self._work_transition["state"][self._index] = tansition.state
    self._work_transition["action"][self._index] = tansition.action
    self._work_transition["reward"][self._index] = tansition.reward

    # エピソード終了
    if tansition.done:
      self._index = 0
      self._count = 0

      return self.get_data()
    else:
      self._count += 1

      # データ追加用に末尾を空ける
      self._work_transition[:-1] = self._work_transition[1:]

      # エピソード開始後、バッファが埋まっていない
      if self._index < self._seq_len - 1:
        self._index += 1
      else:
        return self.get_data()
