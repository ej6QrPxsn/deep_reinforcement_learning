

from typing import NamedTuple
import numpy as np

from config import Config


class EnvOutput(NamedTuple):
  next_state: np.ndarray
  reward: np.ndarray
  done: np.ndarray


class Env:
  def __init__(self, env_name) -> None:
    self.action_space = None

  def reset(self):
    pass

  def step(self, action) -> EnvOutput:
    pass


class BatchedEnv(Env):
  def __init__(self, config: Config, create_env_func) -> None:
    self.envs = np.zeros(config.n_env_batches, dtype=object)
    for i in range(config.n_env_batches):
      self.envs[i] = create_env_func(config.env_name, config.max_timestep)

    self.next_states = np.empty((config.n_env_batches, *config.state_shape), dtype=np.uint8)
    self.rewards = np.zeros(config.n_env_batches)
    self.dones = np.zeros(config.n_env_batches, dtype=bool)

  def step(self, actions) -> EnvOutput:
    for i, (env, action) in enumerate(zip(self.envs, actions)):
      output = env.step(action)
      self.next_states[i] = output.next_state
      self.rewards[i] = output.reward
      self.dones[i] = output.done

    return EnvOutput(
        next_state=self.next_states,
        reward=self.rewards,
        done=self.dones,
    )

  def reset(self):
    for i, env in enumerate(self.envs):
      self.next_states[i] = env.reset().next_state

    self.rewards[:] = 0
    self.dones[:] = False

    return EnvOutput(
        next_state=self.next_states,
        reward=self.rewards,
        done=self.dones,
    )
