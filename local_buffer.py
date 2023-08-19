import numpy as np
from config import Config
from env import EnvOutput
from utils import AgentInputData, SelectActionOutput


class LocalBuffer:
  def __init__(self, batch_size, config: Config) -> None:
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
        ('hidden_state', 'f4', (config.lstm_num_layers, config.lstm_state_size)),
        ('cell_state', 'f4', (config.lstm_num_layers, config.lstm_state_size)),
    ])

    agent_input_dtype = np.dtype([
        ('state', 'u1', (1, *config.state_shape)),
        ('hidden_state', 'f4', (config.lstm_num_layers, config.lstm_state_size)),
        ('cell_state', 'f4', (config.lstm_num_layers, config.lstm_state_size)),
    ])

    self.work_transition = np.zeros((batch_size, config.seq_len + 1), dtype=work_transition_dtype)
    self.all_ids = np.arange(batch_size)
    self.indexes = np.zeros(batch_size, dtype=int)

    self.transition = np.zeros(batch_size, dtype=config.transition_dtype)

    # 次の推論入力用
    self.agent_input = np.zeros(batch_size, dtype=agent_input_dtype)

  def get_agent_input(self):
    return AgentInputData(
        self.agent_input["state"].copy(),
        self.agent_input["hidden_state"].copy(),
        self.agent_input["cell_state"].copy(),
    )

  def add(self, prev_input: AgentInputData, select_action_output: SelectActionOutput, batched_env_output: EnvOutput, betas, gammas):
    self.work_transition["state"][self.all_ids, self.indexes] = prev_input.state[self.all_ids, 0]
    self.work_transition["action"][self.all_ids, self.indexes] = select_action_output.action
    self.work_transition["reward"][self.all_ids, self.indexes] = batched_env_output.reward
    self.work_transition["qvalue"][self.all_ids, self.indexes] = select_action_output.qvalue
    self.work_transition["policy"][self.all_ids, self.indexes] = select_action_output.policy
    self.work_transition["done"][self.all_ids, self.indexes] = batched_env_output.done
    self.work_transition["beta"][self.all_ids, self.indexes] = betas
    self.work_transition["gamma"][self.all_ids, self.indexes] = gammas
    self.work_transition["hidden_state"][self.all_ids, self.indexes] = prev_input.hidden_state
    self.work_transition["cell_state"][self.all_ids, self.indexes] = prev_input.cell_state
    self.indexes += 1

    # 次の推論入力用
    self.agent_input["state"][self.all_ids, 0] = batched_env_output.next_state
    self.agent_input["hidden_state"] = select_action_output.hidden_state
    self.agent_input["cell_state"] = select_action_output.cell_state

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
      self.transition["hidden_state"][:full_id_size] = self.work_transition["hidden_state"][full_ids, 0].copy()
      self.transition["cell_state"][:full_id_size] = self.work_transition["cell_state"][full_ids, 0].copy()

      self.work_transition[full_ids, :self.config.replay_period] = self.work_transition[full_ids, -self.config.replay_period:]
      self.indexes[full_ids] = self.config.replay_period

      ret = (self.transition[:full_id_size], self.work_transition["qvalue"][full_ids].copy())

    done_ids = np.where(batched_env_output.done)[0]
    if done_ids.size > 0:
      self.indexes[done_ids] = 0
      self.agent_input["hidden_state"] = 0
      self.agent_input["cell_state"] = 0

    return ret
