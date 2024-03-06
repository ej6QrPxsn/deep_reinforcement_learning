from typing import List, NamedTuple
import numpy as np
from config import Config
from data_type import DataType


class Transition(NamedTuple):
  state: np.ndarray
  action: np.uint8
  reward: np.float32
  done: bool


class EpisodeTransition(NamedTuple):
  states: List[np.ndarray]
  actions: List[np.uint8]
  rtgs: List[np.float32]
  timestep: int


class LocalBuffer:
  def __init__(self, config: Config, data_type: DataType) -> None:
    self._config = config

    self._episode_steps = 0

    self._states = []
    self._actions = []
    self._rewards = []

  def add_data(self, tansition: Transition):
    self._states += [tansition.state]
    self._actions += [tansition.action]
    self._rewards += [tansition.reward]

    self._episode_steps += 1

    # エピソード終了
    if tansition.done or self._episode_steps >= self._config.max_timestep:
      rtg = np.zeros(len(self._rewards), dtype=np.float32)
      # [1 2 3 4 5 6]
      # [1 3 6 10 15 21]
      # 報酬を足し合わせる
      cumsum_rewards = np.cumsum(self._rewards)
      # return to go [21 20 18 15 11 6]
      rtg[0] = cumsum_rewards[-1]
      rtg[1:] = rtg[0] - cumsum_rewards[:-1]

      data = EpisodeTransition(
        states=self._states,
        actions=self._actions,
        rtgs=rtg.tolist(),
        timestep=self._episode_steps,
      )

      self._episode_steps = 0

      self._states = []
      self._actions = []
      self._rewards = []

      return data
