import numpy as np
import torch
from agent import ActionPredictionNetwork, R2D2Agent, RNDNetwork
from batched_layer import BatchedLayer
from config import Config
from data_type import AgentInputData, DataType, SelectActionOutput
from local_buffer import LocalBuffer
from models import Agent57Network
from reward_generator import RewardGenerator
from shared_data import SharedActorData
from torch.utils.tensorboard import SummaryWriter
import zstandard as zstd

from utils import get_agent_input_burn_in_from_transition, get_beta_table, get_gamma_table, get_input_for_compute_loss, retrace_loss


def eval_loop(shared_infer_net, predict_net, embedding_net, config: Config,
              shared_actor_data: SharedActorData, device):
  shared_actor_data.shared.get_shared_memory()
  batch_size = 1

  RND_net = RNDNetwork(device, config, predict_net)
  ids = [0]
  beta_table = get_beta_table(config)
  rng = np.random.default_rng()  # or random.default_rng(0)

  infer_net = Agent57Network(device, config)
  infer_net.set_weight(*shared_infer_net.get_weight())

  reward_generator = RewardGenerator(batch_size, config, RND_net, embedding_net, device)

  e_prev_lstm_states, i_prev_lstm_states = infer_net.initial_state(batch_size)
  prev_actions = np.zeros((batch_size, 1), dtype=np.uint8)

  _, actor_output = shared_actor_data.get_actor_data()
  agent_input_data = AgentInputData(
    state=np.expand_dims(actor_output.next_state, axis=1),
    prev_action=prev_actions.reshape(batch_size, 1),
    e_prev_reward=actor_output.reward.reshape(batch_size, 1, 1),
    i_prev_reward=actor_output.reward.reshape(batch_size, 1, 1),
    meta_index=actor_output.meta_index.reshape(batch_size, 1),
    e_lstm_states=e_prev_lstm_states,
    i_lstm_states=i_prev_lstm_states,
  )

  beta = torch.from_numpy(beta_table[agent_input_data.meta_index]).to(device)
  select_action_output: SelectActionOutput = infer_net.select_actions(
    agent_input_data, config.eval_epsilon, beta, batch_size)
  shared_actor_data.put_action(select_action_output.action[0])

  prev_actions = select_action_output.action
  e_prev_lstm_states = select_action_output.e_lstm_states
  i_prev_lstm_states = select_action_output.i_lstm_states

  episode_count = 0

  while True:
    # 内部報酬
    intrinsic_rewards = reward_generator.get_intrinsic_reward(ids, agent_input_data.state)

    _, actor_output = shared_actor_data.get_actor_data()
    agent_input_data = AgentInputData(
      state=np.expand_dims(actor_output.next_state, axis=1),
      prev_action=prev_actions.reshape(batch_size, 1),
      e_prev_reward=actor_output.reward.reshape(batch_size, 1, 1),
      i_prev_reward=intrinsic_rewards.reshape(batch_size, 1, 1),
      meta_index=actor_output.meta_index.reshape(batch_size, 1),
      e_lstm_states=e_prev_lstm_states,
      i_lstm_states=i_prev_lstm_states,
    )

    beta = torch.from_numpy(beta_table[agent_input_data.meta_index]).to(device)
    select_action_output: SelectActionOutput = infer_net.select_actions(
      agent_input_data, config.eval_epsilon, beta, batch_size)
    shared_actor_data.put_action(select_action_output.action[0])

    if actor_output.done:
      episode_count += 1
      e_prev_lstm_states, i_prev_lstm_states = infer_net.initial_state(batch_size)
      prev_actions = rng.integers(config.action_space, size=len(ids))
      prev_actions = np.zeros((batch_size, 1), dtype=np.uint8)
      reward_generator.reset(ids)

      if episode_count % config.eval_update_period == 0:
        infer_net.set_weight(*shared_infer_net.get_weight())

    else:
      prev_actions = select_action_output.action
      e_prev_lstm_states = select_action_output.e_lstm_states
      i_prev_lstm_states = select_action_output.i_lstm_states


def inference_loop(actor_indexes, shared_infer_net, predict_net, embedding_net, transition_queue, shared_actor_datas, device, config: Config):
  data_type = DataType(config)
  cctx = zstd.ZstdCompressor(write_content_size=data_type.transition_dtype.itemsize)

  RND_net = RNDNetwork(device, config, predict_net)

  # この推論プロセスで使う環境IDリストを得る
  env_ids = np.arange(
    actor_indexes[0] * config.num_env_batches,
    actor_indexes[-1] * config.num_env_batches + config.num_env_batches
  )
  first_env_id = env_ids[0]

  batch_size = len(env_ids)

  infer_net = Agent57Network(device, config)
  infer_net.set_weight(*shared_infer_net.get_weight())

  reward_generator = RewardGenerator(batch_size, config, RND_net, embedding_net, device)

  beta_table = get_beta_table(config)
  gamma_table = get_gamma_table(config)

  for shared_actor_data in shared_actor_datas:
    shared_actor_data.shared.get_shared_memory()

  local_buffer = LocalBuffer(batch_size, config, reward_generator)
  batched_layer = BatchedLayer(env_ids, shared_actor_datas, config)

  e_prev_lstm_states, i_prev_lstm_states = infer_net.initial_state(batch_size)
  prev_actions = np.zeros(batch_size, dtype=np.uint8)
  prev_meta_indexes = np.zeros(batch_size, dtype=int)

  batched_actor_output = batched_layer.wait_actor_outputs(first_env_id)

  # batched_env_output.next_state(t),
  # meta_indexes(t),
  # batched_env_output.reward(t - 1),
  # batched_env_output.done(t - 1)

  # prev_actions(t - 1),
  # prev_hidden_states(t - 1)
  # prev_cell_states(t - 1)

  agent_input_data = AgentInputData(
    state=np.expand_dims(batched_actor_output.next_state, axis=1),
    prev_action=prev_actions.reshape(batch_size, 1),
    e_prev_reward=batched_actor_output.reward.reshape(batch_size, 1, 1),
    i_prev_reward=batched_actor_output.reward.reshape(batch_size, 1, 1),
    meta_index=batched_actor_output.meta_index.reshape(batch_size, 1),
    e_lstm_states=e_prev_lstm_states,
    i_lstm_states=i_prev_lstm_states,
  )

  # agent_input_data.state(t),
  # agent_input_data.prev_action(t - 1),
  # agent_input_data.e_prev_reward(t - 1),
  # agent_input_data.i_prev_reward(t - 1),
  # agent_input_data.hidden_state(t - 1),
  # agent_input_data.cell_state(t - 1),

  beta = torch.from_numpy(beta_table[agent_input_data.meta_index]).to(device)
  select_action_output: SelectActionOutput = infer_net.select_actions(
    agent_input_data, beta_table[batched_actor_output.meta_index], beta, batch_size)

  # select_action_output.action(t)
  # select_action_output.qvalue(t)
  # select_action_output.policy(t)
  # select_action_output.hidden_states(t)
  # select_action_output.cell_states(t)

  batched_layer.send_actions(select_action_output.action)

  steps = 0
  while True:
    steps += 1

    prev_meta_indexes[:] = batched_actor_output.meta_index
    batched_actor_output = batched_layer.wait_actor_outputs(first_env_id)
    # batched_env_output.next_state(t + 1),
    # meta_indexes(t + 1),
    # prev_meta_indexes(t),
    # batched_env_output.reward(t),
    # batched_env_output.done(t)

    ret = local_buffer.add(agent_input_data, select_action_output, batched_actor_output, prev_meta_indexes)
    if ret and transition_queue.qsize() < config.num_train_envs:
      transitions, qvalues = ret
      qvalues = torch.from_numpy(qvalues[:, config.replay_period:]).to(torch.float32).to(device)

      input = get_input_for_compute_loss(transitions, config, device, beta_table, gamma_table)

      losses = retrace_loss(
        input=input,
        behaviour_qvalues=qvalues,
        target_qvalues=qvalues,
        target_policy_qvalue=qvalues,
        config=config,
        device=device)

      losses = losses.cpu().detach().numpy().copy()

      #: replay_bufferに遷移情報を蓄積
      for loss, transition in zip(losses, transitions):
        transition_queue.put((loss, cctx.compress(transition.tobytes())))

    agent_input_data = local_buffer.get_agent_input()

    beta = torch.from_numpy(beta_table[agent_input_data.meta_index]).to(device)
    select_action_output: SelectActionOutput = infer_net.select_actions(
      agent_input_data, beta_table[batched_actor_output.meta_index], beta, batch_size)

    # select_action_output.action(t + 1)
    # select_action_output.hidden_states(t + 1)
    # select_action_output.cell_states(t + 1)

    # 選択アクションをアクターに送信
    batched_layer.send_actions(select_action_output.action)

    if steps % config.infer_update_period == 0:
      infer_net.set_weight(*shared_infer_net.get_weight())


def train_loop(rank, infer_net, predict_net, embedding_net, sample_queue, priority_queue, device, config: Config):
  if rank == 0:
    summary_writer = SummaryWriter("logs")

  e_weight, i_weight = infer_net.get_weight()

  e_agent = R2D2Agent(device, config)
  e_agent.set_weight(e_weight, e_weight)

  i_agent = R2D2Agent(device, config)
  i_agent.set_weight(i_weight, i_weight)

  RND_net = RNDNetwork(device, config, predict_net)
  action_prediction_net = ActionPredictionNetwork(device, config, embedding_net)

  beta_table = get_beta_table(config)
  gamma_table = get_gamma_table(config)

  steps = 0

  while True:
    #: リプレイバッファからミニバッチを取得
    indexes, transitions, is_weights = sample_queue.get()

    e_input, i_input = get_agent_input_burn_in_from_transition(transitions, device, config)

    e_online_qvalue, e_target_qvalue = e_agent.get_qvalues(e_input, transitions)
    i_online_qvalue, i_target_qvalue = i_agent.get_qvalues(i_input, transitions)

    loss_input = get_input_for_compute_loss(transitions, config, device, beta_table, gamma_table)

    # リプレイのための統一損失
    beta = loss_input.beta.unsqueeze(-1)
    online_qvalue = e_online_qvalue + beta * i_online_qvalue
    target_qvalue = e_target_qvalue + beta * i_target_qvalue
    losses = retrace_loss(loss_input, online_qvalue, target_qvalue, target_qvalue, config, device)

    priority_queue.put((indexes, losses.cpu().detach().numpy()))

    loss = (torch.FloatTensor(is_weights).to(device) * losses).mean(0)

    # 訓練のための個別損失
    e_losses = retrace_loss(loss_input, e_online_qvalue, e_target_qvalue, target_qvalue, config, device)
    i_losses = retrace_loss(loss_input, i_online_qvalue, i_target_qvalue, target_qvalue, config, device)

    # seq sum -> batch mean
    e_loss = (torch.FloatTensor(is_weights).to(device) * e_losses).mean(0)
    i_loss = (torch.FloatTensor(is_weights).to(device) * i_losses).mean(0)

    # 訓練
    action_prediction_net.train(transitions)
    RND_net.train(transitions)
    e_agent.train(e_loss)
    i_agent.train(i_loss)

    del transitions

    # 推論モデル更新
    if rank == 0:
      summary_writer.add_scalar("loss", loss, steps)
      infer_net.set_weight(e_agent.online_net.get_weight(), i_agent.online_net.get_weight())

    steps += 1

    #: target-QネットワークをQネットワークと同期
    if steps % config.target_update_period == 0:
      e_agent.update_target()
      i_agent.update_target()
