

import numpy as np

from config import Config
from env import EnvOutput


class BatchedLayer:
  def __init__(self, env_ids, shared_env_datas, config: Config) -> None:
    self.shared_env_datas = shared_env_datas
    self.config = config
    self.batch_size = len(env_ids)
    self.next_states = np.empty((self.batch_size, *config.state_shape))
    self.rewards = np.zeros(self.batch_size)
    self.transition_dones = np.zeros(self.batch_size, dtype=bool)
    self.dones = np.zeros(self.batch_size, dtype=bool)
    self.indexes = np.arange(self.batch_size).reshape(-1, config.num_env_batches)
    self.betas = np.zeros(self.batch_size)
    self.gammas = np.zeros(self.batch_size)

  def send_actions(self, actions):
    for indexes, shared_env_data in zip(self.indexes, self.shared_env_datas):
      shared_env_data.put_action(actions[indexes])

  def wait_states(self, first_env_id):
    for shared_env_data in self.shared_env_datas:
      ids, states, betas = shared_env_data.get_states()
      indexes = ids - first_env_id
      self.next_states[indexes] = states
      self.betas[indexes] = betas

    return self.next_states, self.betas

  def wait_env_outputs(self, first_env_id):
    for shared_env_data in self.shared_env_datas:
      ids, env_outputs, betas, gammas = shared_env_data.get_env_data()
      indexes = ids - first_env_id
      self.next_states[indexes] = env_outputs.next_state
      self.rewards[indexes] = env_outputs.reward
      self.transition_dones[indexes] = env_outputs.transition_done
      self.dones[indexes] = env_outputs.done
      self.betas[indexes] = betas
      self.gammas[indexes] = gammas

    return EnvOutput(
        next_state=self.next_states,
        reward=self.rewards,
        done=self.dones,
        transition_done=self.transition_dones
    ), self.betas, self.gammas
