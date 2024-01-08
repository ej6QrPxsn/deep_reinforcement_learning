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
    self._index = 0
    self._data_queue = data_queue

    self._work_transition = np.zeros(self._seq_len, dtype=data_type.work_transition_dtype)

  def add(self, state, reward, action, done):
    self._work_transition["state"][self._index] = state
    self._work_transition["action"][self._index] = action
    self._work_transition["reward"][self._index] = reward

    # 蓄積長さ
    if self._index == self._seq_len - 1:
      # 報酬を逆順に
      reverse_rewards = self._work_transition["reward"][::-1]
      # 足し合わせて逆順にすると、最後までの報酬合計が得られる
      total_rewards = np.cumsum(reverse_rewards)[::-1]

      transition = Transition(
        state=self._work_transition["state"],
        rtg=total_rewards,
        action=self._work_transition["action"],
        timestep=self._index
      )

      self._data_queue.put(transition)

      # データ追加用に末尾を空ける
      self._work_transition[:-1] = self._work_transition[1:]

    else:
      self._index += 1

    # エピソード終了
    if done:
      self._index = 0
