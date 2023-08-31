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
  beta: torch.Tensor
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
  rewards = transitions["extrinsic_reward"][:, config.replay_period:] + transitions["beta"][:, config.replay_period:] * transitions["intrinsic_reward"][:, config.replay_period:]
  return ComputeLossInput(
      action=torch.from_numpy(transitions["action"][:, config.replay_period:].copy()).to(torch.int64).unsqueeze(-1).to(device),
      reward=torch.from_numpy(rewards.copy()).to(torch.float32).to(device),
      done=torch.from_numpy(transitions["done"][:, config.replay_period:].copy()).to(torch.bool).to(device),
      policy=torch.from_numpy(transitions["policy"][:, config.replay_period:].copy()).unsqueeze(-1).to(torch.float32).to(device),
      beta=torch.from_numpy(transitions["beta"][:, config.replay_period:].copy()).unsqueeze(-1).to(torch.float32).to(device),
      gamma=torch.from_numpy(transitions["gamma"][:, config.replay_period:].copy()).to(torch.float32).to(device),
  )


def h(x):
  """Value function rescaling per R2D2 paper, table 2."""
  eps = 1e-3
  return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + eps * x


def h_1(x):
  """See Proposition A.2 in paper "Observe and Look Further"."""
  eps = 1e-3
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


def select_actions(agent_output, n_actions, epsilons, device, num_envs):
  tensor_epsilons = torch.from_numpy(epsilons).to(torch.float32).reshape(-1, 1, 1).to(device)
  qvalue, (hidden_state, cell_state) = agent_output

  greedy_action, greedy_policy, epsilon_greedy_policy = get_epsilon_greedy_policy(qvalue, n_actions, tensor_epsilons, device)
  random_actions = rng.integers(n_actions, size=num_envs)
  probs = rng.random(num_envs)

  actions = np.where(probs < epsilons,
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


def get_td(rewards, dones, target_policies, target_qvalues, target_q, gammas):
  return rewards[:, :-1] + gammas[:, 1:] * torch.sum(target_policies[:, 1:] * target_qvalues[:, 1:], 2) - target_q[:, :-1]


def get_trace_coefficients(actions, past_greedy_policies, target_policies, prevent_division_by_zero_tensor, one_tensor, retrace_lambda):
  # ゼロ除算防止
  no_zero_past_greedy_policies = torch.fmax(past_greedy_policies, prevent_division_by_zero_tensor)

  # 重点サンプリングの割合を取得
  policy_rates = target_policies.gather(2, actions) / no_zero_past_greedy_policies

  # batch
  is_action_rate = retrace_lambda * torch.fmin(policy_rates, one_tensor)
  return is_action_rate.squeeze(-1)


def get_retrace_operator(s, trace_coefficients, td, target_q, gammas, n_steps):
  ret = [torch.pow(gammas[:, j], j) * trace_coefficients[:, s + 1:j + 1].prod(1) * td[:, j] for j in range(s, n_steps)]
  return target_q[:, s] + torch.stack(ret).sum(0)


def retrace_loss(input: ComputeLossInput, behaviour_qvalues, target_qvalues, config: Config, device):
  # tdのj+1のため、実質的なseqは-1
  seq_len = behaviour_qvalues.shape[1] - 1

  prevent_division_by_zero_tensor = torch.tensor(config.epsilon, device=device)
  one_tensor = torch.tensor(1, device=device)

  _, _, target_policies = get_epsilon_greedy_policy(target_qvalues, config.action_space, input.beta, device)

  behaviour_q = behaviour_qvalues.gather(2, input.action).squeeze(-1)

  # batch, seq
  target_q = target_qvalues.gather(2, input.action).squeeze(-1)

  td = get_td(input.reward, input.done, target_policies, target_qvalues, target_q, input.gamma)

  trace_coefficients = get_trace_coefficients(input.action, input.policy, target_policies, prevent_division_by_zero_tensor, one_tensor, config.retrace_lambda)

  losses = torch.stack([(behaviour_q[:, s] - get_retrace_operator(s, trace_coefficients, td, target_q, input.gamma, seq_len)) ** 2 for s in range(seq_len)])
  # seq, batch -> batch
  return losses.sum(0)
