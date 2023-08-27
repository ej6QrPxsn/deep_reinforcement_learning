import numpy as np
from config import Config
from numpy import random


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class MetaController:
  def __init__(self, config: Config) -> None:
    self._config = config
    self._reward_table = np.zeros((config.num_env_batches, config.bandit_window_size, config.num_arms))
    self._betas = np.zeros((config.num_env_batches, config.num_arms))
    self._gammas = np.zeros((config.num_env_batches, config.num_arms))
    self._arm_index = np.zeros((config.num_env_batches, 1), dtype=int)
    self._all_ids = np.arange(config.num_env_batches)
    for i in range(config.num_arms):
      self._betas[:, i] = self._get_beta(i)
      self._gammas[:, i] = self._get_gamma(i)

    self._rng = random.default_rng()

  def reset(self):
    self._reward_table[self._all_ids] = 0
    self._arm_index[self._all_ids] = self._config.num_arms - 1
    beta = np.take_along_axis(self._betas, self._arm_index, axis=1)
    gamma = np.take_along_axis(self._gammas, self._arm_index, axis=1)
    return beta.squeeze(-1), gamma.squeeze(-1)

  def update(self, ids, steps, rewards):
    self._reward_table[ids, 1:, :] = self._reward_table[ids, 0:-1, :]
    self._reward_table[ids, 0, self._arm_index[ids]] = rewards[ids]

    self._arm_index[ids] = self._get_arm_index(ids, steps)[:, np.newaxis]
    beta = np.take_along_axis(self._betas, self._arm_index, axis=1)
    gamma = np.take_along_axis(self._gammas, self._arm_index, axis=1)
    return beta.squeeze(-1), gamma.squeeze(-1)

  def _get_max_arg_index(self, ids, steps):
    steps = np.where(steps < 2, 2, steps)
    window_size = self._config.bandit_window_size
    np_window_size = np.empty(len(ids))
    np_window_size[:] = window_size
    arm_mean_rewards = np.sum(self._reward_table[ids, -window_size:], axis=1) / window_size
    param = self._config.bandit_UCB_beta * np.sqrt(np.log(np.min([steps[ids] - 1, np_window_size], axis=0)) / window_size)
    return np.argmax(arm_mean_rewards + param[:, np.newaxis], axis=1)

  def _get_arm_index(self, ids, steps):
    batch_size = len(ids)
    indexes = np.where(steps[ids] < self._config.num_arms, self._config.num_arms - steps[ids] - 1,
                       np.where(self._rng.random(batch_size) >= self._config.bandit_epsilon,
                                self._get_max_arg_index(ids, steps),
                                self._rng.integers(0, self._config.num_arms - 1, size=batch_size))
                       )
    return indexes

  def _get_beta(self, index):
    config = self._config
    if index == 0:
      return 0
    elif index == config.num_arms - 1:
      return config.epsilon_beta
    else:
      return config.epsilon_beta * sigmoid(10 * (2 * index - (config.num_arms - 2)) / (config.num_arms - 2))

  def _get_gamma(self, index):
    config = self._config
    gmax = (config.num_arms - 1 - index) * np.log(1 - config.gamma_max)
    gmin = index * np.log(1 - config.gamma_min)
    return 1 - np.exp((gmax + gmin) / (config.num_arms - 1))
