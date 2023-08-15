import math
import numpy as np
from config import Config
from numpy import random


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class MetaController:
  def __init__(self, config: Config) -> None:
    self.config = config
    self.reward_table = np.zeros((config.num_env_batches, config.bandit_window_size, config.num_arms))
    self.arm_beta = np.zeros((config.num_env_batches, config.num_arms))
    self.arm_gamma = np.zeros((config.num_env_batches, config.num_arms))
    self.arm_index = np.zeros((config.num_env_batches, 1), dtype=int)
    self.all_ids = np.arange(config.num_env_batches)
    for i in range(config.num_arms):
      self.arm_beta[:, i] = self.get_beta(i)
      self.arm_gamma[:, i] = self.get_gamma(i)

    self.steps = 0
    self.rng = random.default_rng()
    self.lower_num_arms = np.zeros((config.num_env_batches, 1), dtype=int)

  def reset(self):
    self.steps = 0
    self.reward_table[:] = 0
    self.arm_index[:] = 0
    beta = np.take_along_axis(self.arm_beta, self.arm_index, axis=1)
    gamma = np.take_along_axis(self.arm_gamma, self.arm_index, axis=1)
    return beta.squeeze(-1), gamma.squeeze(-1)

  def update(self, rewards):
    self.steps += 1
    self.reward_table[:, 1:, :] = self.reward_table[:, 0:-1, :]
    self.reward_table[self.all_ids, 0, self.arm_index[self.all_ids]] = rewards

    self.arm_index[:] = self.get_arm_index()
    beta = np.take_along_axis(self.arm_beta, self.arm_index, axis=1)
    gamma = np.take_along_axis(self.arm_gamma, self.arm_index, axis=1)
    return beta.squeeze(-1), gamma.squeeze(-1)

  def arg_max_index(self):
    windows_size = self.config.bandit_window_size
    arm_mean_rewards = np.sum(self.reward_table[:, -windows_size:], axis=1) / windows_size
    param = self.config.bandit_UCB_beta * math.sqrt(math.log(min(self.steps - 1, windows_size)) / windows_size)
    return np.argmax(arm_mean_rewards + param, axis=1)

  def get_arm_index(self):
    if self.steps < self.config.num_arms:
      self.lower_num_arms[:] = self.steps
      return self.lower_num_arms

    argmax = self.arg_max_index()
    rand = self.rng.integers(0, self.config.num_arms - 1, size=self.config.num_env_batches)
    return np.where(self.rng.random(self.config.num_env_batches) >= self.config.bandit_epsilon,
                    argmax,
                    rand).reshape(-1, 1)

  def get_beta(self, index):
    config = self.config
    if index == 0:
      return 0
    elif index == config.num_arms - 1:
      return config.epsilon_beta
    else:
      return config.epsilon_beta * sigmoid(10 * (2 * index - (config.num_arms - 2)) / (config.num_arms - 2))

  def get_gamma(self, index):
    config = self.config
    gmax = (config.num_arms - 1 - index) * np.log(1 - config.gamma_max)
    gmin = index * np.log(1 - config.gamma_min)
    return 1 - np.exp((gmax + gmin) / (config.num_arms - 1))
