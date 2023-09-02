from typing import NamedTuple, Tuple
import numpy as np
import torch
from numpy import random
from config import Config
rng = random.default_rng()  # or random.default_rng(0)


class AgentInputData(NamedTuple):
  state: np.ndarray
  prev_action: np.ndarray
  prev_extrinsic_reward: np.ndarray
  prev_intrinsic_reward: np.ndarray
  beta: np.ndarray
  hidden_state: np.ndarray
  cell_state: np.ndarray


class SelectActionOutput(NamedTuple):
  action: np.ndarray
  qvalue: np.ndarray
  policy: np.ndarray
  hidden_state: np.ndarray
  cell_state: np.ndarray


class AgentInput(NamedTuple):
  state: torch.Tensor
  prev_action: torch.Tensor
  prev_extrinsic_reward: torch.Tensor
  prev_intrinsic_reward: torch.Tensor
  beta: torch.Tensor
  prev_lstm_state: Tuple[torch.Tensor, torch.Tensor]


class AgentOutput(NamedTuple):
  qvalue: torch.Tensor
  hidden_state: torch.Tensor
  cell_state: torch.Tensor


class ComputeLossInput(NamedTuple):
  action: torch.Tensor
  reward: torch.Tensor
  done: torch.Tensor
  policy: torch.Tensor
  gamma: torch.Tensor


def to_agent_input(agent_input_data: AgentInputData, device) -> AgentInput:
  return AgentInput(
      state=torch.from_numpy(agent_input_data.state.copy()).to(torch.float32).to(device),
      prev_action=torch.from_numpy(agent_input_data.prev_action.copy()).to(torch.int64).to(device),
      prev_extrinsic_reward=torch.from_numpy(agent_input_data.prev_extrinsic_reward.copy()).to(torch.float32).to(device),
      prev_intrinsic_reward=torch.from_numpy(agent_input_data.prev_intrinsic_reward.copy()).to(torch.float32).to(device),
      beta=torch.from_numpy(agent_input_data.beta.copy()).to(torch.float32).to(device),
      prev_lstm_state=(
        # batch, num_layer -> num_layer, batch
        torch.from_numpy(agent_input_data.hidden_state.copy()).permute(1, 0, 2).to(device),
        torch.from_numpy(agent_input_data.cell_state.copy()).permute(1, 0, 2).to(device)
      )
  )


def get_input_for_compute_loss(transitions, config: Config, device) -> ComputeLossInput:
  rewards = transitions["extrinsic_reward"][:, config.replay_period:] + transitions["beta"][:, np.newaxis] * transitions["intrinsic_reward"][:, config.replay_period:]
  return ComputeLossInput(
      action=torch.from_numpy(transitions["action"][:, config.replay_period:].copy()).to(torch.int64).unsqueeze(-1).to(device),
      reward=torch.from_numpy(rewards.copy()).to(torch.float32).to(device),
      done=torch.from_numpy(transitions["done"][:, config.replay_period:].copy()).to(torch.bool).to(device),
      policy=torch.from_numpy(transitions["policy"][:, config.replay_period:].copy()).unsqueeze(-1).to(torch.float32).to(device),
      gamma=torch.from_numpy(transitions["gamma"][:, np.newaxis].copy()).to(torch.float32).to(device),
  )


# rescaling
def h(x, eps):
  return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + eps * x


# inverse rescaling
def h_1(x, eps):
  return torch.sign(x) * (((torch.sqrt(1. + 4 * eps * (torch.abs(x) + 1. + eps)) - 1.) / 2. * eps) - 1.)


def get_epsilon_greedy_policy(qvalues, n_actions, epsilon, device):
  greedy_action = torch.argmax(qvalues, 2, keepdim=True)

  epsilon_greedy_policy = torch.zeros(qvalues.shape).to(device)
  # 1 - epsilonの確率でq値が最大のものだけ選択される
  # (1 - epsilon)
  epsilon_greedy_policy.scatter_(2, greedy_action, 1 - epsilon)
  # epsilonの確率でランダムアクション
  # epsilon * (1 / アクション数)
  epsilon_greedy_policy[:] += epsilon / n_actions

  greedy_policy = epsilon_greedy_policy.gather(2, greedy_action)
  # 選択されるアクション、ポリシー、すべてのポリシーを返す
  return greedy_action, greedy_policy, epsilon_greedy_policy


def select_actions(agent_output, betas, device, config, batch_size):
  tensor_epsilons = torch.empty((batch_size, 1, 1), dtype=torch.float32, device=device)
  tensor_epsilons[:] = torch.tensor(betas).reshape(-1, 1, 1)

  qvalue, (hidden_state, cell_state) = agent_output

  # ポリシー取得
  greedy_action, greedy_policy, epsilon_greedy_policy = get_epsilon_greedy_policy(qvalue, config.action_space, tensor_epsilons, device)
  random_actions = rng.integers(config.action_space, size=batch_size)
  probs = rng.random(batch_size)

  # アクション取得
  actions = np.where(probs < betas,
                     random_actions,
                     # batch, seq, value -> batch
                     greedy_action.reshape(-1).cpu().detach().numpy().copy())

  # 選択されるアクションとポリシーを返す
  return SelectActionOutput(
    action=actions,
    qvalue=qvalue.squeeze(1).cpu().detach().numpy().copy(),
    policy=greedy_policy.reshape(-1).cpu().detach().numpy().copy(),
    # num_layer, batch -> batch, num_layer
    hidden_state=hidden_state.permute(1, 0, 2).cpu().detach().numpy().copy(),
    cell_state=cell_state.permute(1, 0, 2).cpu().detach().numpy().copy()
  )


def get_td(rewards, target_policies, target_qvalues, target_q, gammas, eps):
  return rewards[:, :-1] + gammas * torch.sum(target_policies[:, 1:] * h_1(target_qvalues[:, 1:], eps), 2) - h_1(target_q[:, :-1], eps)


def get_trace_coefficients(actions, past_greedy_policies, target_policies, one_tensor, retrace_lambda):
  # ゼロ除算防止
  past_greedy_policies = torch.where(past_greedy_policies == 0, 1, past_greedy_policies)
  # 重点サンプリングの割合を取得
  policy_rates = target_policies.gather(2, actions) / past_greedy_policies

  # batch
  is_action_rate = retrace_lambda * torch.fmin(policy_rates, one_tensor)
  return is_action_rate.squeeze(-1)


def get_retrace_operator(s, trace_coefficients, td, target_q, gammas, seq_len, eps):
  ret = [torch.pow(gammas[:, 0], j) * trace_coefficients[:, s + 1:j + 1].prod(1) * td[:, j] for j in range(s, seq_len)]
  return h(h_1(target_q[:, s], eps) + torch.stack(ret).sum(0), eps)


def retrace_loss(input: ComputeLossInput, behaviour_qvalues, target_qvalues, config: Config, device):
  # tdのj+1のため、実質的なseqは-1
  seq_len = behaviour_qvalues.shape[1] - 1

  one_tensor = torch.tensor(1, device=device)
  target_epsilon = torch.empty((behaviour_qvalues.shape[0], behaviour_qvalues.shape[1], 1), dtype=torch.float32, device=device)
  target_epsilon[:] = config.target_epsilon

  _, _, target_policies = get_epsilon_greedy_policy(target_qvalues, config.action_space, target_epsilon, device)

  behaviour_q = behaviour_qvalues.gather(2, input.action).squeeze(-1)

  # batch, seq
  target_q = target_qvalues.gather(2, input.action).squeeze(-1)

  td = get_td(input.reward, target_policies, target_qvalues, target_q, input.gamma, config.rescaling_epsilon)

  trace_coefficients = get_trace_coefficients(input.action, input.policy, target_policies, one_tensor, config.retrace_lambda)

  losses = torch.stack([(behaviour_q[:, s] - get_retrace_operator(s, trace_coefficients, td, target_q, input.gamma, seq_len, config.rescaling_epsilon)) ** 2 for s in range(seq_len)])

  # seq, batch -> batch
  return losses.sum(0)
