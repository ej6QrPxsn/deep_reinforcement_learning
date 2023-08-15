import numpy as np
import torch
from batched_layer import BatchedLayer
from config import Config
from local_buffer import LocalBuffer
from models import DQNAgent
from shared_data import SharedEnvData
from utils import get_input_for_compute_loss, retrace_loss
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import zstandard as zstd


def eval_loop(infer_model, shared_env_data: SharedEnvData):
  shared_env_data.shared.get_shared_memory()
  while True:
    _, next_states, betas = shared_env_data.get_states()
    action, _, _ = infer_model.select_actions(next_states, betas, 1)
    shared_env_data.put_action(action[0])


def inference_loop(actor_indexes, infer_model, transition_queue, shared_env_datas, device, config: Config):
  cctx = zstd.ZstdCompressor(write_content_size=config.transition_dtype.itemsize)

  # この推論プロセスで使う環境IDリストを得る
  env_ids = np.arange(
    actor_indexes[0] * config.num_env_batches,
    actor_indexes[-1] * config.num_env_batches + config.num_env_batches
  )
  first_env_id = env_ids[0]

  for shared_env_data in shared_env_datas:
    shared_env_data.shared.get_shared_memory()

  local_buffer = LocalBuffer(env_ids, config)
  batched_layer = BatchedLayer(env_ids, shared_env_datas, config)

  next_states, betas = batched_layer.wait_states(first_env_id)

  actions, qvalues, policies = infer_model.select_actions(next_states, betas, len(env_ids))

  batched_layer.send_actions(actions)

  states = next_states.copy()
  while True:
    batched_env_output, betas, gammas = batched_layer.wait_env_outputs(first_env_id)
    next_states = batched_env_output.next_state

    ret = local_buffer.add(states, actions, qvalues, policies, batched_env_output, betas, gammas)
    if ret and transition_queue.qsize() < 10:
      transitions, qvalues = ret
      qvalues = torch.from_numpy(qvalues).to(torch.float32).to(device)

      input = get_input_for_compute_loss(transitions, device)

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

    actions, qvalues, policies = infer_model.select_actions(next_states, betas, len(env_ids))

    batched_layer.send_actions(actions)

    states = next_states.copy()


def train_loop(rank, infer_model, sample_queue, priority_queue, device, config: Config):
  if rank == 0:
    summary_writer = SummaryWriter("logs")

  agent = DQNAgent(device, config.action_space)
  agent.qnet.load_state_dict(infer_model.state_dict())
  agent.target_qnet.load_state_dict(infer_model.state_dict())

  optimizer = optim.Adam(agent.qnet.parameters(), lr=0.00048, eps=config.epsilon)
  steps = 0

  while True:
    #: リプレイバッファからミニバッチを取得
    indexes, transitions, is_weights = sample_queue.get()

    states = torch.from_numpy(transitions["state"]).to(torch.float32).to(device)
    input = get_input_for_compute_loss(transitions, device)

    # batch * seq, value
    state_shape = states.shape
    model_inputs = states.view(-1, *state_shape[2:])

    behaviour_qvalues = agent.qnet(model_inputs).view(state_shape[0], state_shape[1], -1)
    target_qvalues = agent.target_qnet(model_inputs).view(state_shape[0], state_shape[1], -1)

    losses = retrace_loss(input, behaviour_qvalues, target_qvalues, config, device)

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
