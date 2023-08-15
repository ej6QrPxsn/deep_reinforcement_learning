from typing import NamedTuple
import numpy as np
import torch
from numpy import random
from config import Config
rng = random.default_rng()  # or random.default_rng(0)


class InputForComputeLoss(NamedTuple):
  actions: torch.Tensor
  rewards: torch.Tensor
  dones: torch.Tensor
  policies: torch.Tensor
  betas: torch.Tensor
  gammas: torch.Tensor


def get_input_for_compute_loss(transitions, device) -> InputForComputeLoss:
  return InputForComputeLoss(
      actions=torch.from_numpy(transitions["action"].copy()).to(torch.int64).unsqueeze(-1).to(device),
      rewards=torch.from_numpy(transitions["reward"].copy()).to(torch.float32).to(device),
      dones=torch.from_numpy(transitions["done"].copy()).to(torch.bool).to(device),
      policies=torch.from_numpy(transitions["policy"].copy()).unsqueeze(-1).to(torch.float32).to(device),
      betas=torch.from_numpy(transitions["beta"].copy()).unsqueeze(-1).to(torch.float32).to(device),
      gammas=torch.from_numpy(transitions["gamma"].copy()).to(torch.float32).to(device),
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


def select_actions(qvalues, n_actions, epsilons, device, num_envs):
  tensor_epsilons = torch.from_numpy(epsilons).to(torch.float32).reshape(-1, 1, 1).to(device)

  greedy_action, greedy_policy, epsilon_greedy_policy = get_epsilon_greedy_policy(qvalues, n_actions, tensor_epsilons, device)
  random_actions = rng.integers(n_actions, size=num_envs)
  probs = rng.random(num_envs)

  actions = np.where(probs < epsilons,
                     random_actions,
                     # batch, seq, value -> batch
                     greedy_action.reshape(-1).cpu().detach().numpy().copy())

  # 選択されるアクションとポリシーを返す
  return actions, greedy_policy.reshape(-1).cpu().detach().numpy().copy()


def get_td(rewards, dones, target_policies, target_qvalues, target_q, gammas):
  return rewards[:, :-1] + gammas[:, :-1] * ~dones[:, 1:] * torch.sum(target_policies[:, 1:] * target_qvalues[:, 1:], 2) - target_q[:, :-1]


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


def retrace_loss(input: InputForComputeLoss, behaviour_qvalues, target_qvalues, config: Config, device):
  prevent_division_by_zero_tensor = torch.tensor(config.epsilon).to(device)
  one_tensor = torch.tensor(1).to(device)

  _, _, target_policies = get_epsilon_greedy_policy(target_qvalues, config.action_space, input.betas, device)

  behaviour_q = behaviour_qvalues.gather(2, input.actions).squeeze(-1)

  # batch, seq
  target_q = target_qvalues.gather(2, input.actions).squeeze(-1)

  td = get_td(input.rewards, input.dones, target_policies, target_qvalues, target_q, input.gammas)

  trace_coefficients = get_trace_coefficients(input.actions, input.policies, target_policies, prevent_division_by_zero_tensor, one_tensor, config.retrace_lambda)

  losses = torch.stack([(behaviour_q[:, s] - get_retrace_operator(s, trace_coefficients, td, target_q, input.gammas, config.seq_len)) ** 2 for s in range(config.seq_len)])
  # seq, batch -> batch
  return losses.sum(0)
