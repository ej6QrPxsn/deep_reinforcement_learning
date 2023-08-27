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
        ('prev_action', 'u1'),
        ('prev_reward', 'f4'),
        ('prev_hidden_state', 'f4', (config.lstm_num_layers, config.lstm_state_size)),
        ('prev_cell_state', 'f4', (config.lstm_num_layers, config.lstm_state_size)),
    ])

    agent_input_dtype = np.dtype([
        ('state', 'u1', (1, *config.state_shape)),
        ('prev_action', 'u1', (1,)),
        ('prev_reward', 'f4', (1, 1)),
        ('prev_hidden_state', 'f4', (config.lstm_num_layers, config.lstm_state_size)),
        ('prev_cell_state', 'f4', (config.lstm_num_layers, config.lstm_state_size)),
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
        self.agent_input["prev_action"].copy(),
        self.agent_input["prev_reward"].copy(),
        self.agent_input["prev_hidden_state"].copy(),
        self.agent_input["prev_cell_state"].copy(),
    )

  def _to_transition(self, ids):
    id_size = ids.size

    self.transition["state"][:id_size] = self.work_transition["state"][ids]
    self.transition["action"][:id_size] = self.work_transition["action"][ids]
    self.transition["reward"][:id_size] = self.work_transition["reward"][ids]
    self.transition["policy"][:id_size] = self.work_transition["policy"][ids]
    self.transition["done"][:id_size] = self.work_transition["done"][ids]
    self.transition["beta"][:id_size] = self.work_transition["beta"][ids]
    self.transition["gamma"][:id_size] = self.work_transition["gamma"][ids]
    self.transition["prev_action"][:id_size] = self.work_transition["prev_action"][ids, 0]
    self.transition["prev_reward"][:id_size] = self.work_transition["prev_reward"][ids, 0]
    self.transition["prev_hidden_state"][:id_size] = self.work_transition["prev_hidden_state"][ids, 0]
    self.transition["prev_cell_state"][:id_size] = self.work_transition["prev_cell_state"][ids, 0]

    ret = (self.transition[:id_size].copy(), self.work_transition["qvalue"][ids].copy())
    self.transition[:id_size] = 0

    return ret

  def add(self, prev_input: AgentInputData, select_action_output: SelectActionOutput, batched_env_output: EnvOutput, betas, gammas):
    # prev_input.state(t),
    # prev_input.prev_action(t - 1),
    # prev_input.prev_reward(t - 1),
    # prev_input.hidden_state(t - 1),
    # prev_input.cell_state(t - 1),

    # select_action_output.action(t)
    # select_action_output.qvalue(t)
    # select_action_output.policy(t)
    # select_action_output.hidden_states(t)
    # select_action_output.cell_states(t)

    # batched_env_output.next_state(t + 1),
    # batched_env_output.reward(t),
    # batched_env_output.done(t)

    self.work_transition["state"][self.all_ids, self.indexes] = prev_input.state[self.all_ids, 0]
    self.work_transition["action"][self.all_ids, self.indexes] = select_action_output.action
    self.work_transition["qvalue"][self.all_ids, self.indexes] = select_action_output.qvalue
    self.work_transition["policy"][self.all_ids, self.indexes] = select_action_output.policy
    self.work_transition["reward"][self.all_ids, self.indexes] = batched_env_output.reward
    self.work_transition["done"][self.all_ids, self.indexes] = batched_env_output.done
    self.work_transition["beta"][self.all_ids, self.indexes] = betas
    self.work_transition["gamma"][self.all_ids, self.indexes] = gammas

    self.work_transition["prev_action"][self.all_ids, self.indexes] = prev_input.prev_action[:, 0]
    self.work_transition["prev_reward"][self.all_ids, self.indexes] = prev_input.prev_reward[:, 0, 0]
    self.work_transition["prev_hidden_state"][self.all_ids, self.indexes] = prev_input.hidden_state
    self.work_transition["prev_cell_state"][self.all_ids, self.indexes] = prev_input.cell_state
    self.indexes += 1

    # 次の推論入力用
    self.agent_input["state"][self.all_ids, 0] = batched_env_output.next_state
    self.agent_input["prev_action"][self.all_ids, 0] = select_action_output.action
    self.agent_input["prev_reward"][self.all_ids, 0, 0] = batched_env_output.reward
    self.agent_input["prev_hidden_state"] = select_action_output.hidden_state
    self.agent_input["prev_cell_state"] = select_action_output.cell_state

    ret = ()
    full_ids = np.where(self.indexes > self.config.seq_len)[0]
    if full_ids.size > 0:
      ret = self._to_transition(full_ids)

      self.work_transition[full_ids, :self.config.replay_period] = self.work_transition[full_ids, -self.config.replay_period:]
      self.indexes[full_ids] = self.config.replay_period

    done_ids = np.where(batched_env_output.done)[0]
    if done_ids.size > 0:
      # doneの次以降をクリア
      for done_id in done_ids:
        self.work_transition[done_id, self.indexes[done_id]:] = 0

      ret = self._to_transition(done_ids)

      self.indexes[done_ids] = 0

      self.agent_input["prev_action"][done_ids] = 0
      self.agent_input["prev_reward"][done_ids] = 0
      self.agent_input["prev_hidden_state"][done_ids] = 0
      self.agent_input["prev_cell_state"][done_ids] = 0

    return ret
