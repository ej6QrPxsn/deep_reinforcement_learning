import numpy as np
import torch
from config import Config
from models import R2D2Network, RNDRandomNetwork

from utils import AgentInput, get_input_for_compute_loss, retrace_loss
import torch.optim as optim
import torch.nn as nn


class R2D2Agent:
  def __init__(self, device, config: Config):
    self._device = device
    self._config = config

    self.qnet = R2D2Network(self._device, self._config).to(self._device)
    self.target_qnet = R2D2Network(self._device, self._config).to(self._device)
    self._optimizer = optim.Adam(
      params=self.qnet.parameters(),
      lr=config.adam_r2d2_learning_rate,
      betas=(config.adam_beta1, config.adam_beta2),
      eps=config.adam_epsilon)

  def update_target(self):
    self.target_qnet.load_state_dict(self.qnet.state_dict())

  def get_agent_input_burn_in_from_transition(self, transition):
    prev_action = np.concatenate([
      transition["prev_action"][:, np.newaxis],
      transition["action"][:, :self._config.replay_period - 1]
    ], axis=1)
    prev_extrinsic_reward = np.concatenate([
      transition["prev_extrinsic_reward"][:, np.newaxis],
      transition["extrinsic_reward"][:, :self._config.replay_period - 1]
    ], axis=1)
    prev_intrinsic_reward = np.concatenate([
      transition["prev_intrinsic_reward"][:, np.newaxis],
      transition["intrinsic_reward"][:, :self._config.replay_period - 1]
    ], axis=1)

    policy_index = torch.empty(prev_action.shape, dtype=int, device=self._device)
    policy_index[:] = torch.from_numpy(transition["policy_index"][:, np.newaxis].copy())

    return AgentInput(
      state=torch.from_numpy(transition["state"][:, :self._config.replay_period].copy()).to(torch.float32).to(self._device),
      prev_action=torch.from_numpy(prev_action.copy()).to(torch.int64).to(self._device),
      prev_extrinsic_reward=torch.from_numpy(prev_extrinsic_reward.copy()).unsqueeze(-1).to(torch.float32).to(self._device),
      prev_intrinsic_reward=torch.from_numpy(prev_intrinsic_reward.copy()).unsqueeze(-1).to(torch.float32).to(self._device),
      policy_index=policy_index,
      prev_lstm_state=(
        # batch, num_layer -> num_layer, batch
        torch.from_numpy(transition["prev_hidden_state"].copy()).to(torch.float32).permute(1, 0, 2).to(self._device),
        torch.from_numpy(transition["prev_cell_state"].copy()).to(torch.float32).permute(1, 0, 2).to(self._device)
      )
    )

  def get_agent_input_from_transition(self, transition, lstm_state):
    policy_index = torch.empty(transition["action"][:, self._config.replay_period - 1:-1].shape, dtype=int, device=self._device)
    policy_index[:] = torch.from_numpy(transition["policy_index"][:, np.newaxis].copy())

    return AgentInput(
      state=torch.from_numpy(transition["state"][:, self._config.replay_period:].copy()).to(torch.float32).to(self._device),
      prev_action=torch.from_numpy(transition["action"][:, self._config.replay_period - 1:-1].copy()).to(torch.int64).to(self._device),
      prev_extrinsic_reward=torch.from_numpy(transition["extrinsic_reward"][:, self._config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(self._device),
      prev_intrinsic_reward=torch.from_numpy(transition["intrinsic_reward"][:, self._config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(self._device),
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

    input = get_input_for_compute_loss(transitions, self._config, self._device, beta_table, gamma_table)

    return retrace_loss(input, qnet_out, target_qnet_out, self._config, self._device)

  def train(self, loss):
    self._optimizer.zero_grad()
    loss.backward()

    # 勾配反映
    self._optimizer.step()


class RNDNetwork:

  def __init__(self, device, config: Config, predict, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    self._config = config
    self._device = device

    self._predict = predict

    self._random = RNDRandomNetwork(device, config)
    self._random.to(device)

    self._random.load_state_dict(self._predict.state_dict())

    self._criterion = torch.nn.MSELoss(reduction="none")
    self._optimizer = torch.optim.Adam(
      params=self._predict.parameters(),
      lr=config.adam_rnd_learning_rate,
      betas=(config.adam_beta1, config.adam_beta2),
      eps=config.adam_epsilon)

  def get_loss(self, x):
    rand_out = self._random(x)
    predict_out = self._predict(x)

    loss = self._criterion(predict_out, rand_out)
    return loss.mean(1)

  def train(self, transition):
    state = torch.from_numpy(transition["state"][:, -self._config.embedding_train_period:].copy()).to(torch.float32).to(self._device)

    rand_out = self._random(state.reshape(-1, *state.shape[2:]))
    predict_out = self._predict(state.reshape(-1, *state.shape[2:]))

    loss = self._criterion(predict_out, rand_out)

    self._optimizer.zero_grad()
    loss.mean().backward()
    self._optimizer.step()


class ActionPredictionNetwork:

  def __init__(self, device, config: Config, embedding, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    self._config = config
    self._device = device
    self._embedding = embedding

    self._feature = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, config.action_space),
        nn.Softmax(dim=1)
    )
    self._feature.to(device)

    self._criterion = nn.CrossEntropyLoss()
    self._optimizer = torch.optim.Adam(
      params=self._embedding.parameters(),
      lr=config.adam_action_prediction_learning_rate,
      betas=(config.adam_beta1, config.adam_beta2),
      weight_decay=config.adam_action_prediction_ls_weight,
      eps=config.adam_epsilon)

  def train(self, transition):
    state = torch.from_numpy(transition["state"][:, -self._config.embedding_train_period - 1:-1].copy()).to(torch.float32).to(self._device)
    next_state = torch.from_numpy(transition["state"][:, -self._config.embedding_train_period:].copy()).to(torch.float32).to(self._device)
    action = torch.from_numpy(transition["action"][:, -self._config.embedding_train_period - 1:-1].copy()).to(torch.int64).to(self._device)

    ret1 = self._embedding(state.reshape(-1, *state.shape[2:]))
    ret2 = self._embedding(next_state.reshape(-1, *next_state.shape[2:]))
    out = self._feature(torch.cat([ret1, ret2], dim=1))

    loss = self._criterion(out, action.reshape(-1, *action.shape[2:]))

    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()
