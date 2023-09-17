import numpy as np
from config import Config
from data_type import AgentInputData, DataType, SelectActionOutput
from reward_generator import RewardGenerator
from shared_data import ActorOutput


class LocalBuffer:
  def __init__(self, batch_size, config: Config, reward_generator: RewardGenerator) -> None:
    self._config = config
    self._reward_generator = reward_generator

    # | replay_period | trace_length | 1 |
    #                 |replay_period | trace_length | 1 |
    self._seq_len = config.seq_len + 1
    self._len = config.seq_len + config.trace_length + 1

    data_type = DataType(config)

    self._work_transition = np.zeros((batch_size, self._len), dtype=data_type.work_transition_dtype)
    self._all_ids = np.arange(batch_size)
    self._indexes = np.zeros(batch_size, dtype=int)

    self._transition = np.zeros(batch_size, dtype=data_type.transition_dtype)

    # 損失計算用
    self._loss_qvalues = np.empty((batch_size, self._seq_len, self._config.action_space))

    # 次の推論入力用
    self._agent_input = np.zeros(batch_size, dtype=data_type.agent_input_dtype)

    self._base_indexes = np.zeros(batch_size, dtype=int)

  def get_agent_input(self):
    return AgentInputData(
        self._agent_input["state"].copy(),
        self._agent_input["prev_action"].copy(),
        self._agent_input["e_prev_reward"].copy(),
        self._agent_input["i_prev_reward"].copy(),
        self._agent_input["meta_index"].copy(),
        self._agent_input["prev_hidden_state"].copy(),
        self._agent_input["prev_cell_state"].copy(),
    )

  def add(self, prev_input: AgentInputData, select_action_output: SelectActionOutput,
          batched_actor_output: ActorOutput, prev_meta_indexes):
    # prev_input.state(t)
    # prev_input.prev_action(t - 1)
    # prev_input.e_prev_reward(t - 1)
    # prev_input.i_prev_reward(t - 1)
    # prev_input.hidden_state(t - 1)
    # prev_input.cell_state(t - 1)

    # prev_meta_indexes(t)

    # select_action_output.action(t)
    # select_action_output.qvalue(t)
    # select_action_output.policy(t)
    # select_action_output.hidden_states(t)
    # select_action_output.cell_states(t)

    # batched_env_output.next_state(t + 1)
    # batched_env_output.reward(t)
    # batched_env_output.done(t)

    # 内部報酬
    intrinsic_rewards = self._reward_generator.get_intrinsic_reward(self._all_ids, prev_input.state)

    self._work_transition["state"][self._all_ids, self._indexes] = prev_input.state[self._all_ids, 0]
    self._work_transition["action"][self._all_ids, self._indexes] = select_action_output.action
    self._work_transition["qvalue"][self._all_ids, self._indexes] = select_action_output.qvalue
    self._work_transition["policy"][self._all_ids, self._indexes] = select_action_output.policy
    self._work_transition["e_reward"][self._all_ids, self._indexes] = batched_actor_output.reward
    self._work_transition["i_reward"][self._all_ids, self._indexes] = intrinsic_rewards
    self._work_transition["done"][self._all_ids, self._indexes] = batched_actor_output.done
    self._work_transition["meta_index"][self._all_ids, self._indexes] = prev_meta_indexes

    self._work_transition["prev_action"][self._all_ids, self._indexes] = prev_input.prev_action[:, 0]
    self._work_transition["e_prev_reward"][self._all_ids, self._indexes] = prev_input.e_prev_reward[:, 0, 0]
    self._work_transition["i_prev_reward"][self._all_ids, self._indexes] = prev_input.i_prev_reward[:, 0, 0]
    self._work_transition["prev_hidden_state"][self._all_ids, self._indexes] = prev_input.hidden_state
    self._work_transition["prev_cell_state"][self._all_ids, self._indexes] = prev_input.cell_state
    self._indexes += 1

    # 次の推論入力用
    self._agent_input["state"][self._all_ids, 0] = batched_actor_output.next_state
    self._agent_input["prev_action"][self._all_ids, 0] = select_action_output.action
    self._agent_input["e_prev_reward"][self._all_ids, 0, 0] = batched_actor_output.reward
    self._agent_input["i_prev_reward"][self._all_ids, 0, 0] = intrinsic_rewards
    self._agent_input["meta_index"][self._all_ids, 0] = batched_actor_output.meta_index
    self._agent_input["prev_hidden_state"] = select_action_output.hidden_state
    self._agent_input["prev_cell_state"] = select_action_output.cell_state

    ret = ()
    # 蓄積長さ
    store_length = self._indexes - self._base_indexes
    store_ids = np.where(store_length >= self._seq_len)[0]
    if store_ids.size > 0:
      # シーケンス長さの蓄積があるなら遷移として使う
      ret = self._to_transition(store_ids)

      # エピソード初回の遷移
      first_ids = np.where(self._base_indexes == 0)[0]
      first_ids = list(set(store_ids) & set(first_ids))
      if len(first_ids) > 0:
        self._base_indexes[first_ids] = self._config.seq_len - self._config.replay_period

      # エピソード2回目以降の遷移
      second_ids = np.where(self._base_indexes > 0)[0]
      second_ids = list(set(store_ids) & set(second_ids))
      if len(second_ids) > 0:
        # | replay_period | trace_length | 1 |
        #                | replay_period | trace_length | 1 |
        # | replay_period | trace_length | 1 |
        #                | replay_period |
        #                                    ^ <-index
        self._work_transition[second_ids, :self._seq_len] = self._work_transition[second_ids, -self._seq_len:]
        self._indexes[second_ids] = self._config.seq_len
        # self.work_transition[second_ids, :-1] = self.work_transition[second_ids, 1:]
        # self.indexes[second_ids] -= 1

    # エピソード終了
    done_ids = np.where(batched_actor_output.done)[0]
    if done_ids.size > 0:
      # シーケンス長さの蓄積がある
      store_ids = np.where(self._indexes >= self._seq_len)[0]
      store_ids = list(set(store_ids) & set(done_ids))
      if len(store_ids) > 0:
        ret = self._to_transition(store_ids)

      self._indexes[done_ids] = 0
      self._base_indexes[done_ids] = 0

      self._reward_generator.reset(done_ids)

      # state, meta_indexはエピソード終了で次の値を受け取っているので、ここでは設定しない
      self._agent_input["prev_action"][done_ids] = 0
      self._agent_input["e_prev_reward"][done_ids] = 0
      self._agent_input["i_prev_reward"][done_ids] = 0
      self._agent_input["prev_hidden_state"][done_ids] = 0
      self._agent_input["prev_cell_state"][done_ids] = 0
    return ret

  def _to_transition(self, ids):
    for i, id in enumerate(ids):
      self._transition["state"][i] = self._work_transition["state"][id, self._indexes[id] - self._seq_len:self._indexes[id]]
      self._transition["action"][i] = self._work_transition["action"][id, self._indexes[id] - self._seq_len:self._indexes[id]]
      self._transition["e_reward"][i] = self._work_transition["e_reward"][id, self._indexes[id] - self._seq_len:self._indexes[id]]
      self._transition["i_reward"][i] = self._work_transition["i_reward"][id, self._indexes[id] - self._seq_len:self._indexes[id]]
      self._transition["policy"][i] = self._work_transition["policy"][id, self._indexes[id] - self._seq_len:self._indexes[id]]
      self._transition["done"][i] = self._work_transition["done"][id, self._indexes[id] - self._seq_len:self._indexes[id]]
      self._transition["meta_index"][i] = self._work_transition["meta_index"][id, self._base_indexes[id]]
      self._transition["prev_action"][i] = self._work_transition["prev_action"][id, self._base_indexes[id]]
      self._transition["e_prev_reward"][i] = self._work_transition["e_prev_reward"][id, self._base_indexes[id]]
      self._transition["i_prev_reward"][i] = self._work_transition["i_prev_reward"][id, self._base_indexes[id]]
      self._transition["prev_hidden_state"][i] = self._work_transition["prev_hidden_state"][id, self._base_indexes[id]]
      self._transition["prev_cell_state"][i] = self._work_transition["prev_cell_state"][id, self._base_indexes[id]]
      self._loss_qvalues[i] = self._work_transition["qvalue"][id, self._indexes[id] - self._seq_len:self._indexes[id]]

    ret = (self._transition[:len(ids)].copy(), self._loss_qvalues[:len(ids)].copy())

    return ret
