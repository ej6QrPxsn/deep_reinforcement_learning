import numpy as np
import torch
from batched_layer import BatchedLayer
from config import Config
from local_buffer import LocalBuffer
from models import DQNAgent
from shared_data import SharedEnvData
from utils import AgentInputData, SelectActionOutput, to_agent_input, get_agent_input_burn_in_from_transition, get_agent_input_from_transition, get_input_for_compute_loss, retrace_loss
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import zstandard as zstd


def eval_loop(infer_model, shared_env_data: SharedEnvData, device):
  shared_env_data.shared.get_shared_memory()
  batch_size = 1

  prev_hidden_states, prev_cell_states = infer_model.initial_state(batch_size)
  prev_actions = np.zeros((batch_size, 1), dtype=np.uint8)

  while True:
    _, env_output, betas, _ = shared_env_data.get_env_data()
    agent_input_data = AgentInputData(
      state=np.expand_dims(env_output.next_state, axis=1),
      prev_action=prev_actions.reshape(batch_size, 1),
      prev_reward=env_output.reward.reshape(batch_size, 1, 1),
      hidden_state=prev_hidden_states,
      cell_state=prev_cell_states,
    )
    select_action_output: SelectActionOutput = infer_model.select_actions(
      # seq 1 を追加
      to_agent_input(agent_input_data, device),
      betas, batch_size)
    shared_env_data.put_action(select_action_output.action[0])

    if env_output.done:
      prev_hidden_states, prev_cell_states = infer_model.initial_state(batch_size)
      prev_actions = np.zeros((batch_size, 1), dtype=np.uint8)
    else:
      prev_actions = select_action_output.action
      prev_hidden_states = select_action_output.hidden_state
      prev_cell_states = select_action_output.cell_state


def inference_loop(actor_indexes, infer_model, transition_queue, shared_env_datas, device, config: Config):
  cctx = zstd.ZstdCompressor(write_content_size=config.transition_dtype.itemsize)

  # この推論プロセスで使う環境IDリストを得る
  env_ids = np.arange(
    actor_indexes[0] * config.num_env_batches,
    actor_indexes[-1] * config.num_env_batches + config.num_env_batches
  )
  first_env_id = env_ids[0]

  batch_size = len(env_ids)

  for shared_env_data in shared_env_datas:
    shared_env_data.shared.get_shared_memory()

  local_buffer = LocalBuffer(batch_size, config)
  batched_layer = BatchedLayer(env_ids, shared_env_datas, config)

  prev_hidden_states, prev_cell_states = infer_model.initial_state(batch_size)
  prev_actions = np.zeros(batch_size, dtype=np.uint8)

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
    prev_reward=batched_env_output.reward.reshape(batch_size, 1, 1),
    hidden_state=prev_hidden_states,
    cell_state=prev_cell_states,
  )

  # agent_input_data.state(t),
  # agent_input_data.prev_action(t - 1),
  # agent_input_data.prev_reward(t - 1),
  # agent_input_data.hidden_state(t - 1),
  # agent_input_data.cell_state(t - 1),

  select_action_output: SelectActionOutput = infer_model.select_actions(
    to_agent_input(agent_input_data, device),
    betas, batch_size)

  # select_action_output.action(t)
  # select_action_output.qvalue(t)
  # select_action_output.policy(t)
  # select_action_output.hidden_states(t)
  # select_action_output.cell_states(t)

  batched_layer.send_actions(select_action_output.action)

  while True:
    batched_env_output, betas, gammas = batched_layer.wait_env_outputs(first_env_id)
    # batched_env_output.next_state(t + 1),
    # betas(t + 1),
    # gammas(t + 1),
    # prev_betas(t),
    # prev_gammas(t),
    # batched_env_output.reward(t),
    # batched_env_output.done(t)

    ret = local_buffer.add(agent_input_data, select_action_output, batched_env_output, betas, gammas)
    if ret and transition_queue.qsize() < config.num_train_envs:
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

    agent_input_data = local_buffer.get_agent_input()

    select_action_output: SelectActionOutput = infer_model.select_actions(
      to_agent_input(agent_input_data, device),
      betas, batch_size)

    # select_action_output.action(t + 1)
    # select_action_output.hidden_states(t + 1)
    # select_action_output.cell_states(t + 1)

    # 選択アクションをアクターに送信
    batched_layer.send_actions(select_action_output.action)


def train_loop(rank, infer_model, sample_queue, priority_queue, device, config: Config):
  if rank == 0:
    summary_writer = SummaryWriter("logs")

  agent = DQNAgent(device, config)
  agent.qnet.load_state_dict(infer_model.state_dict())
  agent.target_qnet.load_state_dict(infer_model.state_dict())

  optimizer = optim.Adam(agent.qnet.parameters(), lr=0.00048, eps=config.epsilon)
  steps = 0

  while True:
    #: リプレイバッファからミニバッチを取得
    indexes, transitions, is_weights = sample_queue.get()

    # burn in
    model_input = get_agent_input_burn_in_from_transition(transitions, config, device)
    _, qnet_lstm_state = agent.qnet(model_input)
    _, target_qnet_lstm_state = agent.target_qnet(model_input)

    # 推論
    qnet_input = get_agent_input_from_transition(transitions, qnet_lstm_state, config, device)
    target_qnet_input = get_agent_input_from_transition(transitions, target_qnet_lstm_state, config, device)
    qnet_out, _ = agent.qnet(qnet_input)
    target_qnet_out, _ = agent.target_qnet(target_qnet_input)

    input = get_input_for_compute_loss(transitions, config, device)

    losses = retrace_loss(input, qnet_out, target_qnet_out, config, device)

    losses = torch.nan_to_num(losses)
    priority_queue.put((indexes, losses.cpu().detach().numpy()))

    # seq sum -> batch mean
    loss = (torch.FloatTensor(is_weights).to(device) * losses).mean(0)

    optimizer.zero_grad()
    loss.backward()

    # 勾配反映
    optimizer.step()

    del transitions

    # 推論モデル更新
    if rank == 0:
      summary_writer.add_scalar('loss', loss, steps)
      infer_model.load_state_dict(agent.qnet.state_dict())

    steps += 1

    #: 2500ステップごとにtarget-QネットワークをQネットワークと同期
    if steps % 2500 == 0:
      agent.target_qnet.load_state_dict(agent.qnet.state_dict())
