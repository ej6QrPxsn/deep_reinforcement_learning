import numpy as np
import torch
from agent import R2D2Agent
from batched_layer import BatchedLayer
from config import Config
from local_buffer import LocalBuffer
from models import RNDNetwork
from parallel_task import ParallelTask
from reward_generator import RewardGenerator
from shared_data import SharedEnvData
from utils import AgentInputData, SelectActionOutput, select_actions, to_agent_input, get_input_for_compute_loss, retrace_loss
from torch.utils.tensorboard import SummaryWriter
import zstandard as zstd


def eval_loop(infer_net, RND_predict_net, embedding_net, config: Config,
              shared_env_data: SharedEnvData, device):
  shared_env_data.shared.get_shared_memory()
  batch_size = 1
  ids = [0]

  RND_net = RNDNetwork(device, config, RND_predict_net)
  reward_generator = RewardGenerator(batch_size, config)
  parallel_task = ParallelTask()

  prev_hidden_states, prev_cell_states = infer_net.initial_state(batch_size)
  prev_actions = np.zeros((batch_size, 1), dtype=np.uint8)

  _, env_output, betas, _ = shared_env_data.get_env_data()
  agent_input_data = AgentInputData(
    state=np.expand_dims(env_output.next_state, axis=1),
    prev_action=prev_actions.reshape(batch_size, 1),
    prev_extrinsic_reward=env_output.reward.reshape(batch_size, 1, 1),
    prev_intrinsic_reward=env_output.reward.reshape(batch_size, 1, 1),
    beta=betas.reshape(batch_size, 1, 1),
    hidden_state=prev_hidden_states,
    cell_state=prev_cell_states,
  )

  infer_output, future_reward = parallel_task.inference(
    ids,
    to_agent_input(agent_input_data, device),
    reward_generator, infer_net, RND_net, embedding_net)
  select_action_output: SelectActionOutput = select_actions(infer_output, config.action_space, betas, device, batch_size)

  shared_env_data.put_action(select_action_output.action[0])

  prev_actions = select_action_output.action
  prev_hidden_states = select_action_output.hidden_state
  prev_cell_states = select_action_output.cell_state

  while True:
    # 内部報酬
    intrinsic_rewards = future_reward.result()

    _, env_output, betas, _ = shared_env_data.get_env_data()
    agent_input_data = AgentInputData(
      state=np.expand_dims(env_output.next_state, axis=1),
      prev_action=prev_actions.reshape(batch_size, 1),
      prev_extrinsic_reward=env_output.reward.reshape(batch_size, 1, 1),
      prev_intrinsic_reward=intrinsic_rewards.reshape(batch_size, 1, 1),
      beta=betas.reshape(batch_size, 1, 1),
      hidden_state=prev_hidden_states,
      cell_state=prev_cell_states,
    )
    infer_output, future_reward = parallel_task.inference(
      ids,
      to_agent_input(agent_input_data, device),
      reward_generator, infer_net, RND_net, embedding_net)
    select_action_output: SelectActionOutput = select_actions(infer_output, config.action_space, betas, device, batch_size)

    shared_env_data.put_action(select_action_output.action[0])

    if env_output.done:
      prev_hidden_states, prev_cell_states = infer_net.initial_state(batch_size)
      prev_actions = np.zeros((batch_size, 1), dtype=np.uint8)
      reward_generator.reset(ids)
    else:
      prev_actions = select_action_output.action
      prev_hidden_states = select_action_output.hidden_state
      prev_cell_states = select_action_output.cell_state


def inference_loop(actor_indexes, infer_net, RND_predict_net, embedding_net, transition_queue, shared_env_datas, device, config: Config):
  cctx = zstd.ZstdCompressor(threads=-1)

  # この推論プロセスで使う環境IDリストを得る
  env_ids = np.arange(
    actor_indexes[0] * config.num_env_batches,
    actor_indexes[-1] * config.num_env_batches + config.num_env_batches
  )
  first_env_id = env_ids[0]

  batch_size = len(env_ids)
  ids = env_ids - first_env_id

  RND_net = RNDNetwork(device, config, RND_predict_net)
  reward_generator = RewardGenerator(batch_size, config)
  parallel_task = ParallelTask()

  for shared_env_data in shared_env_datas:
    shared_env_data.shared.get_shared_memory()

  local_buffer = LocalBuffer(batch_size, config, reward_generator)
  batched_layer = BatchedLayer(env_ids, shared_env_datas, config)

  prev_hidden_states, prev_cell_states = infer_net.initial_state(batch_size)
  prev_actions = np.zeros(batch_size, dtype=np.uint8)
  prev_betas = np.zeros(batch_size, dtype=np.float32)
  prev_gammas = np.zeros(batch_size, dtype=np.float32)

  batched_env_output, betas, gammas = batched_layer.wait_env_outputs(first_env_id)

  # batched_env_output.next_state(t),
  # betas(t),
  # gammas(t),
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
    beta=betas.reshape(batch_size, 1, 1),
    hidden_state=prev_hidden_states,
    cell_state=prev_cell_states,
  )

  # agent_input_data.state(t),
  # agent_input_data.prev_action(t - 1),
  # agent_input_data.prev_extrinsic_reward(t - 1),
  # agent_input_data.prev_intrinsic_reward(t - 1),
  # agent_input_data.hidden_state(t - 1),
  # agent_input_data.cell_state(t - 1),

  infer_output, future_reward = parallel_task.inference(
    ids,
    to_agent_input(agent_input_data, device),
    reward_generator, infer_net, RND_net, embedding_net)
  select_action_output: SelectActionOutput = select_actions(infer_output, config.action_space, betas, device, batch_size)

  # select_action_output.action(t)
  # select_action_output.qvalue(t)
  # select_action_output.policy(t)
  # select_action_output.hidden_states(t)
  # select_action_output.cell_states(t)

  batched_layer.send_actions(select_action_output.action)

  while True:
    prev_betas = betas
    prev_gammas = gammas
    batched_env_output, betas, gammas = batched_layer.wait_env_outputs(first_env_id)
    # batched_env_output.next_state(t + 1),
    # betas(t + 1),
    # gammas(t + 1),
    # prev_betas(t),
    # prev_gammas(t),
    # batched_env_output.reward(t),
    # batched_env_output.done(t)

    intrinsic_reward = future_reward.result()
    ret = local_buffer.add(agent_input_data, select_action_output, batched_env_output, prev_betas, prev_gammas, intrinsic_reward)
    if ret and transition_queue.qsize() < config.num_train_envs:
      add_replay(ret, device, config, transition_queue, cctx)

    agent_input_data = local_buffer.get_agent_input()

    infer_output, future_reward = parallel_task.inference(
      ids,
      to_agent_input(agent_input_data, device),
      reward_generator, infer_net, RND_net, embedding_net)

    select_action_output: SelectActionOutput = select_actions(infer_output, config.action_space, betas, device, batch_size)

    # select_action_output.action(t + 1)
    # select_action_output.hidden_states(t + 1)
    # select_action_output.cell_states(t + 1)

    # 選択アクションをアクターに送信
    batched_layer.send_actions(select_action_output.action)


def add_replay(ret, device, config, transition_queue, cctx):
  transitions, qvalues = ret
  qvalues = torch.from_numpy(qvalues[:, config.replay_period:]).to(torch.float32).to(device)

  input = get_input_for_compute_loss(transitions, config, device)

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


def train_loop(rank, infer_net, RND_predict_net, embedding_net, sample_queue, priority_queue, device, config: Config):
  if rank == 0:
    summary_writer = SummaryWriter("logs")

  agent = R2D2Agent(device, config)
  agent.online_net.set_weight(infer_net.get_weight())
  agent.update_target()

  RND_net = RNDNetwork(device, config, RND_predict_net)

  parallel_task = ParallelTask()

  steps = 0

  while True:
    #: リプレイバッファからミニバッチを取得
    indexes, transitions, is_weights = sample_queue.get()

    # 非同期で訓練
    parallel_task.train(transitions, RND_net, embedding_net)

    # 非同期でモデル出力値を取得
    future_online_output, future_target_output = parallel_task.get_agent_output(transitions, agent)

    # 損失計算用
    loss_input = get_input_for_compute_loss(transitions, config, device)

    # 損失計算
    losses = retrace_loss(
      loss_input,
      future_online_output.result(),
      future_target_output.result(),
      config, device)

    priority_queue.put((indexes, losses.cpu().detach().numpy()))

    # seq sum -> batch mean
    loss = (torch.FloatTensor(is_weights, device=device) * losses).mean(0)

    # 訓練
    agent.train(loss)

    del transitions

    # 推論モデル更新
    if rank == 0:
      summary_writer.add_scalar('loss', loss, steps)
      infer_net.set_weight(agent.online_net.get_weight())

    steps += 1

    #: target-QネットワークをQネットワークと同期
    if steps % config.target_update_period == 0:
      agent.update_target()
