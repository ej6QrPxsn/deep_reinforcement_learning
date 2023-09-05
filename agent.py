import numpy as np
import torch
from models import R2D2Network

from utils import AgentInput, get_input_for_compute_loss, retrace_loss
import torch.optim as optim


class R2D2Agent:
  def __init__(self, device, config):
    self.device = device
    self.config = config

    self.qnet = R2D2Network(self.device, self.config).to(self.device)
    self.target_qnet = R2D2Network(self.device, self.config).to(self.device)
    self._optimizer = optim.Adam(self.qnet.parameters(), lr=0.00048, eps=config.adam_epsilon)

  def update_target(self):
    self.target_qnet.load_state_dict(self.qnet.state_dict())

  def get_agent_input_burn_in_from_transition(self, transition):
    prev_action = np.concatenate([
      transition["prev_action"][:, np.newaxis],
      transition["action"][:, :self.config.replay_period - 1]
    ], axis=1)
    prev_extrinsic_reward = np.concatenate([
      transition["prev_extrinsic_reward"][:, np.newaxis],
      transition["extrinsic_reward"][:, :self.config.replay_period - 1]
    ], axis=1)
    prev_intrinsic_reward = np.concatenate([
      transition["prev_intrinsic_reward"][:, np.newaxis],
      transition["intrinsic_reward"][:, :self.config.replay_period - 1]
    ], axis=1)

    policy_index = torch.empty(prev_action.shape, dtype=int, device=self.device)
    policy_index[:] = torch.from_numpy(transition["policy_index"][:, np.newaxis].copy())

    return AgentInput(
      state=torch.from_numpy(transition["state"][:, :self.config.replay_period].copy()).to(torch.float32).to(self.device),
      prev_action=torch.from_numpy(prev_action.copy()).to(torch.int64).to(self.device),
      prev_extrinsic_reward=torch.from_numpy(prev_extrinsic_reward.copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      prev_intrinsic_reward=torch.from_numpy(prev_intrinsic_reward.copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      policy_index=policy_index,
      prev_lstm_state=(
        # batch, num_layer -> num_layer, batch
        torch.from_numpy(transition["prev_hidden_state"].copy()).to(torch.float32).permute(1, 0, 2).to(self.device),
        torch.from_numpy(transition["prev_cell_state"].copy()).to(torch.float32).permute(1, 0, 2).to(self.device)
      )
    )

  def get_agent_input_from_transition(self, transition, lstm_state):
    policy_index = torch.empty(transition["action"][:, self.config.replay_period - 1:-1].shape, dtype=int, device=self.device)
    policy_index[:] = torch.from_numpy(transition["policy_index"][:, np.newaxis].copy())

    return AgentInput(
      state=torch.from_numpy(transition["state"][:, self.config.replay_period:].copy()).to(torch.float32).to(self.device),
      prev_action=torch.from_numpy(transition["action"][:, self.config.replay_period - 1:-1].copy()).to(torch.int64).to(self.device),
      prev_extrinsic_reward=torch.from_numpy(transition["extrinsic_reward"][:, self.config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      prev_intrinsic_reward=torch.from_numpy(transition["intrinsic_reward"][:, self.config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      policy_index=policy_index,
      prev_lstm_state=lstm_state
    )

  def compute_loss(self, transitions, beta_table, gamma_table):
    # burn in
    model_input = self.get_agent_input_burn_in_from_transition(transitions)
    _, qnet_lstm_state = self.qnet(model_input)
    _, target_qnet_lstm_state = self.target_qnet(model_input)

    # 推論
    qnet_input = self.get_agent_input_from_transition(transitions, qnet_lstm_state)
    target_qnet_input = self.get_agent_input_from_transition(transitions, target_qnet_lstm_state)

    qnet_out, _ = self.qnet(qnet_input)
    target_qnet_out, _ = self.target_qnet(target_qnet_input)

    input = get_input_for_compute_loss(transitions, self.config, self.device, beta_table, gamma_table)

    return retrace_loss(input, qnet_out, target_qnet_out, self.config, self.device)

  def train(self, loss):
    self._optimizer.zero_grad()
    loss.backward()

    # 勾配反映
    self._optimizer.step()
