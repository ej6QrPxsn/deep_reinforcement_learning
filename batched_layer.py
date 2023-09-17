

import numpy as np

from config import Config
from shared_data import ActorOutput


class BatchedLayer:
  def __init__(self, env_ids, shared_actor_datas, config: Config) -> None:
    self.shared_actor_datas = shared_actor_datas
    self.config = config
    self.batch_size = len(env_ids)
    self.next_states = np.empty((self.batch_size, *config.state_shape))
    self.rewards = np.zeros(self.batch_size)
    self.dones = np.zeros(self.batch_size, dtype=bool)
    self.indexes = np.arange(self.batch_size).reshape(-1, config.num_env_batches)
    self.meta_indexes = np.zeros(self.batch_size, dtype=int)

  def send_actions(self, actions):
    for indexes, shared_actor_data in zip(self.indexes, self.shared_actor_datas):
      shared_actor_data.put_action(actions[indexes])

  def wait_actor_outputs(self, first_env_id) -> ActorOutput:
    for shared_actor_data in self.shared_actor_datas:
      ids, actor_outputs = shared_actor_data.get_actor_data()
      indexes = ids - first_env_id
      self.next_states[indexes] = actor_outputs.next_state
      self.rewards[indexes] = actor_outputs.reward
      self.dones[indexes] = actor_outputs.done
      self.meta_indexes[indexes] = actor_outputs.meta_index

    return ActorOutput(
        next_state=self.next_states,
        reward=self.rewards,
        done=self.dones,
        meta_index=self.meta_indexes,
    )
