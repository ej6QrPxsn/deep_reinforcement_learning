import numpy as np
import torch
from agent import ActionPredictionNetwork, R2D2Agent, RNDNetwork
from batched_layer import BatchedLayer
from config import Config
from data_type import AgentInputData, DataType, SelectActionOutput
from local_buffer import LocalBuffer
from reward_generator import RewardGenerator
from shared_data import SharedEnvData
from torch.utils.tensorboard import SummaryWriter
import zstandard as zstd

from utils import get_beta_table, get_gamma_table, get_input_for_compute_loss, retrace_loss, to_agent_input


def eval_loop(infer_net, predict_net, embedding_net, config: Config,
              shared_env_data: SharedEnvData, device):
  shared_env_data.shared.get_shared_memory()
  batch_size = 1

  RND_net = RNDNetwork(device, config, predict_net)

  reward_generator = RewardGenerator(batch_size, config, RND_net, embedding_net, device)

  prev_hidden_states, prev_cell_states = infer_net.initial_state(batch_size)
  prev_actions = np.zeros((batch_size, 1), dtype=np.uint8)

  _, env_output, policy_indexes = shared_env_data.get_env_data()
  agent_input_data = AgentInputData(
    state=np.expand_dims(env_output.next_state, axis=1),
    prev_action=prev_actions.reshape(batch_size, 1),
    prev_extrinsic_reward=env_output.reward.reshape(batch_size, 1, 1),
    prev_intrinsic_reward=env_output.reward.reshape(batch_size, 1, 1),
    policy_index=policy_indexes.reshape(batch_size, 1),
    hidden_state=prev_hidden_states,
    cell_state=prev_cell_states,
  )

  select_action_output: SelectActionOutput = infer_net.select_actions(
    to_agent_input(agent_input_data, device),
    config.eval_epsilon, batch_size)
  shared_env_data.put_action(select_action_output.action[0])

  prev_actions = select_action_output.action
  prev_hidden_states = select_action_output.hidden_state
  prev_cell_states = select_action_output.cell_state

  while True:
    # 内部報酬
    intrinsic_rewards = reward_generator.get_intrinsic_reward([0], agent_input_data.state)

    _, env_output, policy_indexes = shared_env_data.get_env_data()
    agent_input_data = AgentInputData(
      state=np.expand_dims(env_output.next_state, axis=1),
      prev_action=prev_actions.reshape(batch_size, 1),
      prev_extrinsic_reward=env_output.reward.reshape(batch_size, 1, 1),
      prev_intrinsic_reward=intrinsic_rewards.reshape(batch_size, 1, 1),
      policy_index=policy_indexes.reshape(batch_size, 1),
      hidden_state=prev_hidden_states,
      cell_state=prev_cell_states,
    )
    select_action_output: SelectActionOutput = infer_net.select_actions(
      # seq 1 を追加
      to_agent_input(agent_input_data, device),
      config.eval_epsilon, batch_size)
    shared_env_data.put_action(select_action_output.action[0])

    if env_output.done:
      prev_hidden_states, prev_cell_states = infer_net.initial_state(batch_size)
      prev_actions = np.zeros((batch_size, 1), dtype=np.uint8)
      reward_generator.reset([0])
    else:
      prev_actions = select_action_output.action
      prev_hidden_states = select_action_output.hidden_state
      prev_cell_states = select_action_output.cell_state


def inference_loop(actor_indexes, infer_net, predict_net, embedding_net, transition_queue, shared_env_datas, device, config: Config):
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

  reward_generator = RewardGenerator(batch_size, config, RND_net, embedding_net, device)

  beta_table = get_beta_table(config)
  gamma_table = get_gamma_table(config)

  for shared_env_data in shared_env_datas:
    shared_env_data.shared.get_shared_memory()

  local_buffer = LocalBuffer(batch_size, config, reward_generator)
  batched_layer = BatchedLayer(env_ids, shared_env_datas, config)

  prev_hidden_states, prev_cell_states = infer_net.initial_state(batch_size)
  prev_actions = np.zeros(batch_size, dtype=np.uint8)
  prev_policy_indexes = np.zeros(batch_size, dtype=int)

  batched_env_output, policy_indexes = batched_layer.wait_env_outputs(first_env_id)

  # batched_env_output.next_state(t),
  # policy_indexes(t),
  # batched_env_output.reward(t - 1),
  # batched_env_output.done(t - 1)

  # prev_actions(t - 1),
  # prev_hidden_states(t - 1)
  # prev_cell_states(t - 1)

  agent_input_data = AgentInputData(
    state=np.expand_dims(batched_env_output.next_state, axis=1),
    prev_action=prev_actions.reshape(batch_size, 1),
    prev_extrinsic_reward=batched_env_output.reward.reshape(batch_size, 1, 1),
    prev_intrinsic_reward=batched_env_output.reward.reshape(batch_size, 1, 1),
    policy_index=policy_indexes.reshape(batch_size, 1),
    hidden_state=prev_hidden_states,
    cell_state=prev_cell_states,
  )

  # agent_input_data.state(t),
  # agent_input_data.prev_action(t - 1),
  # agent_input_data.prev_extrinsic_reward(t - 1),
  # agent_input_data.prev_intrinsic_reward(t - 1),
  # agent_input_data.hidden_state(t - 1),
  # agent_input_data.cell_state(t - 1),

  select_action_output: SelectActionOutput = infer_net.select_actions(
    to_agent_input(agent_input_data, device),
    beta_table[policy_indexes], batch_size)

  # select_action_output.action(t)
  # select_action_output.qvalue(t)
  # select_action_output.policy(t)
  # select_action_output.hidden_states(t)
  # select_action_output.cell_states(t)

  batched_layer.send_actions(select_action_output.action)

  while True:
    prev_policy_indexes[:] = policy_indexes
    batched_env_output, policy_indexes = batched_layer.wait_env_outputs(first_env_id)
    # batched_env_output.next_state(t + 1),
    # policy_indexes(t + 1),
    # prev_policy_indexes(t),
    # batched_env_output.reward(t),
    # batched_env_output.done(t)

    ret = local_buffer.add(agent_input_data, select_action_output, batched_env_output, prev_policy_indexes, policy_indexes)
    if ret and transition_queue.qsize() < config.num_train_envs:
      transitions, qvalues = ret
      qvalues = torch.from_numpy(qvalues[:, config.replay_period:]).to(torch.float32).to(device)

      input = get_input_for_compute_loss(transitions, config, device, beta_table, gamma_table)

      losses = retrace_loss(
        input=input,
        behaviour_qvalues=qvalues,
        target_qvalues=qvalues,
        config=config,
        device=device)

      losses = losses.cpu().detach().numpy().copy()

      #: replay_bufferに遷移情報を蓄積
      for loss, transition in zip(losses, transitions):
        transition_queue.put((loss, cctx.compress(transition.tobytes())))

    agent_input_data = local_buffer.get_agent_input()

    select_action_output: SelectActionOutput = infer_net.select_actions(
      to_agent_input(agent_input_data, device),
      beta_table[policy_indexes], batch_size)

    # select_action_output.action(t + 1)
    # select_action_output.hidden_states(t + 1)
    # select_action_output.cell_states(t + 1)

    # 選択アクションをアクターに送信
    batched_layer.send_actions(select_action_output.action)


def train_loop(rank, infer_net, predict_net, embedding_net, sample_queue, priority_queue, device, config: Config):
  if rank == 0:
    summary_writer = SummaryWriter("logs")

  agent = R2D2Agent(device, config)
  agent.qnet.load_state_dict(infer_net.state_dict())
  agent.target_qnet.load_state_dict(infer_net.state_dict())

  RND_net = RNDNetwork(device, config, predict_net)
  action_prediction_net = ActionPredictionNetwork(device, config, embedding_net)

  beta_table = get_beta_table(config)
  gamma_table = get_gamma_table(config)

  steps = 0

  while True:
    #: リプレイバッファからミニバッチを取得
    indexes, transitions, is_weights = sample_queue.get()

    losses = agent.compute_loss(transitions, beta_table, gamma_table)

    priority_queue.put((indexes, losses.cpu().detach().numpy()))

    # seq sum -> batch mean
    loss = (torch.FloatTensor(is_weights).to(device) * losses).mean(0)

    # 訓練
    action_prediction_net.train(transitions)
    RND_net.train(transitions)
    agent.train(loss)

    del transitions

    # 推論モデル更新
    if rank == 0:
      summary_writer.add_scalar("loss", loss, steps)
      infer_net.load_state_dict(agent.qnet.state_dict())

    steps += 1

    #: target-QネットワークをQネットワークと同期
    if steps % config.target_update_period == 0:
      agent.update_target()
