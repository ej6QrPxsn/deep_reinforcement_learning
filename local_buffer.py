import numpy as np
from config import Config
from data_type import AgentInputData, DataType, SelectActionOutput
from env import EnvOutput
from reward_generator import RewardGenerator


class LocalBuffer:
  def __init__(self, batch_size, config: Config, reward_generator: RewardGenerator) -> None:
    self.config = config
    self.reward_generator = reward_generator

    # | replay_period | trace_length | 1 |
    #                 |replay_period | trace_length | 1 |
    self.seq_len = config.seq_len + 1
    self.len = config.seq_len + config.trace_length + 1

    data_type = DataType(config)

    self.work_transition = np.zeros((batch_size, self.len), dtype=data_type.work_transition_dtype)
    self.all_ids = np.arange(batch_size)
    self.indexes = np.zeros(batch_size, dtype=int)

    self.transition = np.zeros(batch_size, dtype=data_type.transition_dtype)

    # 損失計算用
    self.loss_qvalues = np.empty((batch_size, self.seq_len, self.config.action_space))

    # 次の推論入力用
    self.agent_input = np.zeros(batch_size, dtype=data_type.agent_input_dtype)

    self.base_indexes = np.zeros(batch_size, dtype=int)

  def get_agent_input(self):
    return AgentInputData(
        self.agent_input["state"].copy(),
        self.agent_input["prev_action"].copy(),
        self.agent_input["prev_extrinsic_reward"].copy(),
        self.agent_input["prev_intrinsic_reward"].copy(),
        self.agent_input["policy_index"].copy(),
        self.agent_input["prev_hidden_state"].copy(),
        self.agent_input["prev_cell_state"].copy(),
    )

  def _to_transition(self, ids):
    for i, id in enumerate(ids):
      self.transition["state"][i] = self.work_transition["state"][id, self.indexes[id] - self.seq_len:self.indexes[id]]
      self.transition["action"][i] = self.work_transition["action"][id, self.indexes[id] - self.seq_len:self.indexes[id]]
      self.transition["extrinsic_reward"][i] = self.work_transition["extrinsic_reward"][id, self.indexes[id] - self.seq_len:self.indexes[id]]
      self.transition["intrinsic_reward"][i] = self.work_transition["intrinsic_reward"][id, self.indexes[id] - self.seq_len:self.indexes[id]]
      self.transition["policy"][i] = self.work_transition["policy"][id, self.indexes[id] - self.seq_len:self.indexes[id]]
      self.transition["done"][i] = self.work_transition["done"][id, self.indexes[id] - self.seq_len:self.indexes[id]]
      self.transition["policy_index"][i] = self.work_transition["policy_index"][id, self.base_indexes[id]]
      self.transition["prev_action"][i] = self.work_transition["prev_action"][id, self.base_indexes[id]]
      self.transition["prev_extrinsic_reward"][i] = self.work_transition["prev_extrinsic_reward"][id, self.base_indexes[id]]
      self.transition["prev_intrinsic_reward"][i] = self.work_transition["prev_intrinsic_reward"][id, self.base_indexes[id]]
      self.transition["prev_hidden_state"][i] = self.work_transition["prev_hidden_state"][id, self.base_indexes[id]]
      self.transition["prev_cell_state"][i] = self.work_transition["prev_cell_state"][id, self.base_indexes[id]]
      self.loss_qvalues[i] = self.work_transition["qvalue"][id, self.indexes[id] - self.seq_len:self.indexes[id]]

    ret = (self.transition[:len(ids)].copy(), self.loss_qvalues[:len(ids)].copy())

    return ret

  def add(self, prev_input: AgentInputData, select_action_output: SelectActionOutput,
          batched_env_output: EnvOutput, prev_policy_indexes, policy_indexes):
    # prev_input.state(t)
    # prev_input.prev_action(t - 1)
    # prev_input.prev_extrinsic_reward(t - 1)
    # prev_input.prev_intrinsic_reward(t - 1)
    # prev_input.hidden_state(t - 1)
    # prev_input.cell_state(t - 1)

    # prev_policy_indexes(t)

    # select_action_output.action(t)
    # select_action_output.qvalue(t)
    # select_action_output.policy(t)
    # select_action_output.hidden_states(t)
    # select_action_output.cell_states(t)

    # batched_env_output.next_state(t + 1)
    # batched_env_output.reward(t)
    # batched_env_output.done(t)

    # 内部報酬
    intrinsic_rewards = self.reward_generator.get_intrinsic_reward(self.all_ids, prev_input.state)

    self.work_transition["state"][self.all_ids, self.indexes] = prev_input.state[self.all_ids, 0]
    self.work_transition["action"][self.all_ids, self.indexes] = select_action_output.action
    self.work_transition["qvalue"][self.all_ids, self.indexes] = select_action_output.qvalue
    self.work_transition["policy"][self.all_ids, self.indexes] = select_action_output.policy
    self.work_transition["extrinsic_reward"][self.all_ids, self.indexes] = batched_env_output.reward
    self.work_transition["intrinsic_reward"][self.all_ids, self.indexes] = intrinsic_rewards
    self.work_transition["done"][self.all_ids, self.indexes] = batched_env_output.done
    self.work_transition["policy_index"][self.all_ids, self.indexes] = prev_policy_indexes

    self.work_transition["prev_action"][self.all_ids, self.indexes] = prev_input.prev_action[:, 0]
    self.work_transition["prev_extrinsic_reward"][self.all_ids, self.indexes] = prev_input.prev_extrinsic_reward[:, 0, 0]
    self.work_transition["prev_intrinsic_reward"][self.all_ids, self.indexes] = prev_input.prev_intrinsic_reward[:, 0, 0]
    self.work_transition["prev_hidden_state"][self.all_ids, self.indexes] = prev_input.hidden_state
    self.work_transition["prev_cell_state"][self.all_ids, self.indexes] = prev_input.cell_state
    self.indexes += 1

    # 次の推論入力用
    self.agent_input["state"][self.all_ids, 0] = batched_env_output.next_state
    self.agent_input["prev_action"][self.all_ids, 0] = select_action_output.action
    self.agent_input["prev_extrinsic_reward"][self.all_ids, 0, 0] = batched_env_output.reward
    self.agent_input["prev_intrinsic_reward"][self.all_ids, 0, 0] = intrinsic_rewards
    self.agent_input["policy_index"][self.all_ids, 0] = policy_indexes
    self.agent_input["prev_hidden_state"] = select_action_output.hidden_state
    self.agent_input["prev_cell_state"] = select_action_output.cell_state

    ret = ()
    # 蓄積長さ
    store_length = self.indexes - self.base_indexes
    store_ids = np.where(store_length >= self.seq_len)[0]
    if store_ids.size > 0:
      # シーケンス長さの蓄積があるなら遷移として使う
      ret = self._to_transition(store_ids)

      # エピソード初回の遷移
      first_ids = np.where(self.base_indexes == 0)[0]
      first_ids = list(set(store_ids) & set(first_ids))
      if len(first_ids) > 0:
        self.base_indexes[first_ids] = self.config.seq_len - self.config.replay_period

      # エピソード2回目以降の遷移
      second_ids = np.where(self.base_indexes > 0)[0]
      second_ids = list(set(store_ids) & set(second_ids))
      if len(second_ids) > 0:
        # | replay_period | trace_length | 1 |
        #                | replay_period | trace_length | 1 |
        # | replay_period | trace_length | 1 |
        #                | replay_period |
        #                                    ^ <-index
        self.work_transition[second_ids, :self.seq_len] = self.work_transition[second_ids, -self.seq_len:]
        self.indexes[second_ids] = self.config.seq_len
        # self.work_transition[second_ids, :-1] = self.work_transition[second_ids, 1:]
        # self.indexes[second_ids] -= 1

    # エピソード終了
    done_ids = np.where(batched_env_output.done)[0]
    if done_ids.size > 0:
      # シーケンス長さの蓄積がある
      store_ids = np.where(self.indexes >= self.seq_len)[0]
      store_ids = list(set(store_ids) & set(done_ids))
      if len(store_ids) > 0:
        ret = self._to_transition(store_ids)

      self.indexes[done_ids] = 0
      self.base_indexes[done_ids] = 0

      self.reward_generator.reset(done_ids)

      # state, policy_indexはエピソード終了で次の値を受け取っているので、ここでは設定しない
      self.agent_input["prev_action"][done_ids] = 0
      self.agent_input["prev_extrinsic_reward"][done_ids] = 0
      self.agent_input["prev_intrinsic_reward"][done_ids] = 0
      self.agent_input["prev_hidden_state"][done_ids] = 0
      self.agent_input["prev_cell_state"][done_ids] = 0

    return ret
