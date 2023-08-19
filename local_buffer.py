import numpy as np
from config import Config
from utils import SelectActionOutput


class LocalBuffer:
  def __init__(self, env_ids, config: Config) -> None:
    self.config = config
    work_transition_dtype = np.dtype([
        ('state', 'u1', config.state_shape),
        ('action', 'u1'),
        ('reward', 'f4'),
        ('done', '?'),
        ('policy', 'f4'),
        ('qvalue', 'f4', config.action_space),
        ('beta', 'f4'),
        ('gamma', 'f4'),
    ])

    self.work_transition = np.zeros((len(env_ids), config.seq_len + 1), dtype=work_transition_dtype)
    self.all_ids = np.arange(len(env_ids))
    self.indexes = np.zeros(len(env_ids), dtype=int)

    self.transition = np.zeros(len(env_ids), dtype=config.transition_dtype)

  def add(self, states, select_action_output: SelectActionOutput, batched_env_output, betas, gammas):
    self.work_transition["state"][self.all_ids, self.indexes] = states
    self.work_transition["action"][self.all_ids, self.indexes] = select_action_output.action
    self.work_transition["reward"][self.all_ids, self.indexes] = batched_env_output.reward
    self.work_transition["qvalue"][self.all_ids, self.indexes] = select_action_output.qvalue
    self.work_transition["policy"][self.all_ids, self.indexes] = select_action_output.policy
    self.work_transition["done"][self.all_ids, self.indexes] = batched_env_output.done
    self.work_transition["beta"][self.all_ids, self.indexes] = betas
    self.work_transition["gamma"][self.all_ids, self.indexes] = gammas
    self.indexes += 1

    ret = ()
    full_ids = np.where(self.indexes > self.config.seq_len)[0]
    if full_ids.size > 0:
      full_id_size = full_ids.size

      self.transition["state"][:full_id_size] = self.work_transition["state"][full_ids].copy()
      self.transition["action"][:full_id_size] = self.work_transition["action"][full_ids].copy()
      self.transition["reward"][:full_id_size] = self.work_transition["reward"][full_ids].copy()
      self.transition["policy"][:full_id_size] = self.work_transition["policy"][full_ids].copy()
      self.transition["done"][:full_id_size] = self.work_transition["done"][full_ids].copy()
      self.transition["beta"][:full_id_size] = self.work_transition["beta"][full_ids].copy()
      self.transition["gamma"][:full_id_size] = self.work_transition["gamma"][full_ids].copy()

      self.indexes[full_ids] = self.config.seq_len
      self.work_transition[full_ids, :-1] = self.work_transition[full_ids, 1:]

      ret = (self.transition[:full_id_size], self.work_transition["qvalue"][full_ids].copy())

    done_ids = np.where(batched_env_output.done)[0]
    if done_ids.size > 0:
      self.indexes[done_ids] = 0

    return ret
