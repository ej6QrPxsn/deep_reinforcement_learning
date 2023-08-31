import numpy as np
import torch
from models import R2D2Network

from utils import AgentInput, get_input_for_compute_loss, retrace_loss
import torch.optim as optim


class R2D2Agent:
  def __init__(self, device, config):
    self.device = device
    self.config = config

    self.online_net = R2D2Network(self.device, self.config).to(self.device)
    self.target_net = R2D2Network(self.device, self.config).to(self.device)
    self._optimizer = optim.Adam(self.online_net.parameters(), lr=0.00048, eps=config.epsilon)

  def update_target(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

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

    return AgentInput(
      state=torch.from_numpy(transition["state"][:, :self.config.replay_period].copy()).to(torch.float32).to(self.device),
      prev_action=torch.from_numpy(prev_action.copy()).to(torch.int64).to(self.device),
      prev_extrinsic_reward=torch.from_numpy(prev_extrinsic_reward.copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      prev_intrinsic_reward=torch.from_numpy(prev_intrinsic_reward.copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      beta=torch.from_numpy(transition["beta"][:, :self.config.replay_period].copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      prev_lstm_state=(
        # batch, num_layer -> num_layer, batch
        torch.from_numpy(transition["prev_hidden_state"].copy()).to(torch.float32).permute(1, 0, 2).to(self.device),
        torch.from_numpy(transition["prev_cell_state"].copy()).to(torch.float32).permute(1, 0, 2).to(self.device)
      )
    )

  def get_agent_input_from_transition(self, transition, lstm_state):
    return AgentInput(
      state=torch.from_numpy(transition["state"][:, self.config.replay_period:].copy()).to(torch.float32).to(self.device),
      prev_action=torch.from_numpy(transition["action"][:, self.config.replay_period - 1:-1].copy()).to(torch.int64).to(self.device),
      prev_extrinsic_reward=torch.from_numpy(transition["extrinsic_reward"][:, self.config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      prev_intrinsic_reward=torch.from_numpy(transition["intrinsic_reward"][:, self.config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      beta=torch.from_numpy(transition["beta"][:, self.config.replay_period:].copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      prev_lstm_state=lstm_state
    )

  def get_online_output(self, transitions):
    model_input = self.get_agent_input_burn_in_from_transition(transitions)
    _, online_lstm_state = self.online_net(model_input)
    online_input = self.get_agent_input_from_transition(transitions, online_lstm_state)
    online_output, _ = self.online_net(online_input)
    return online_output

  def get_target_output(self, transitions):
    model_input = self.get_agent_input_burn_in_from_transition(transitions)
    _, target_lstm_state = self.target_net(model_input)
    target_input = self.get_agent_input_from_transition(transitions, target_lstm_state)
    target_output, _ = self.target_net(target_input)
    return target_output

  def train(self, loss):
    self._optimizer.zero_grad()
    loss.backward()

    # 勾配反映
    self._optimizer.step()
