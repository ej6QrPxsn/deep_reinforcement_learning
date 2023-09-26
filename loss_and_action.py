

import numpy as np
import torch

from config import Config
from data_type import ComputeLossInput


def h(x):
  """Value function rescaling per R2D2 paper, table 2."""
  eps = 1e-3
  return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + eps * x


def h_1(x):
  """See Proposition A.2 in paper "Observe and Look Further"."""
  eps = 1e-3
  return torch.sign(x) * (((torch.sqrt(1. + 4 * eps * (torch.abs(x) + 1. + eps)) - 1.) / 2. * eps) - 1.)


class LossAndAction:
  def __init__(self, qvalue_shape, device, config: Config) -> None:
    self._device = device
    self._config = config
    self._one_tensor = torch.tensor(1, device=device)
    self._target_epsilon = torch.empty((qvalue_shape[0], qvalue_shape[1], 1), dtype=torch.float32, device=device)
    self._epsilon_greedy_policy = torch.zeros(qvalue_shape, device=device)
    self._rng = np.random.default_rng()

  def get_epsilon_greedy_policy(self, qvalues):
    greedy_action = torch.argmax(qvalues, 2, keepdim=True)

    greedy_probability = 1 - self._target_epsilon * ((self._config.action_space - 1) / self._config.action_space)

    # epsilonの確率でランダムアクション
    self._epsilon_greedy_policy[:] = self._target_epsilon / self._config.action_space

    # 1 - epsilonの確率でq値が最大のものだけ選択される
    self._epsilon_greedy_policy.scatter_(2, greedy_action, greedy_probability)

    greedy_policy = self._epsilon_greedy_policy.gather(2, greedy_action)
    # 選択されるアクション、ポリシー、すべてのポリシーを返す
    return greedy_action, greedy_policy, self._epsilon_greedy_policy

  def select_actions(self, qvalues, epsilons, batch_size):
    self._target_epsilon[:] = torch.tensor(epsilons).reshape(-1, 1, 1)

    greedy_action, greedy_policy, _ = self.get_epsilon_greedy_policy(qvalues)
    random_actions = self._rng.integers(self._config.action_space, size=batch_size)
    probs = self._rng.random(batch_size)

    actions = np.where(probs < epsilons,
                       random_actions,
                       # batch, seq, value -> batch
                       greedy_action.reshape(-1).cpu().detach().numpy().copy())

    # 選択されるアクションとポリシーを返す
    return actions, greedy_policy.reshape(-1).cpu().detach().numpy().copy()

  def get_td(self, rewards, target_policies, target_qvalues, target_q, gammas):
    return rewards[:, :-1].clone() + gammas * torch.sum(target_policies[:, 1:].clone() * target_qvalues[:, 1:].clone(), 2) - target_q[:, :-1].clone()

  def get_trace_coefficients(self, actions, past_greedy_policies, target_policies):
    # ゼロ除算防止
    past_greedy_policies = torch.where(past_greedy_policies == 0, 1, past_greedy_policies)
    # 重点サンプリングの割合を取得
    policy_rates = target_policies.gather(2, actions) / past_greedy_policies

    # batch
    is_action_rate = self._config.retrace_lambda * torch.fmin(policy_rates, self._one_tensor)
    return is_action_rate.squeeze(-1)

  def get_retrace_operator(self, s, trace_coefficients, td, target_q, gammas, seq_len):
    ret = [torch.pow(gammas[:, 0], j) * trace_coefficients[:, s + 1:j + 1].prod(1) * td[:, j] for j in range(s, seq_len)]
    return target_q[:, s] + torch.stack(ret).sum(0)

  def retrace_loss(self, input: ComputeLossInput, behaviour_qvalues, target_qvalues, target_policy_qvalue):
    # tdのj+1のため、実質的なseqは-1
    seq_len = behaviour_qvalues.shape[1] - 1

    self._target_epsilon[:] = self._config.target_epsilon

    _, _, target_policies = self.get_epsilon_greedy_policy(target_policy_qvalue)

    behaviour_q = behaviour_qvalues.gather(2, input.action).squeeze(-1)

    # batch, seq
    target_q = target_qvalues.gather(2, input.action).squeeze(-1)

    td = self.get_td(input.reward, target_policies, target_qvalues, target_q, input.gamma)

    trace_coefficients = self.get_trace_coefficients(input.action, input.policy, target_policies)

    losses = torch.stack([(behaviour_q[:, s] - self.get_retrace_operator(s, trace_coefficients, td, target_q, input.gamma, seq_len)) ** 2 for s in range(seq_len)])
    # seq, batch -> batch
    return losses.sum(0)
