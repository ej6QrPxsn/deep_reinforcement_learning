import torch
from config import Config
from data_type import AgentInput
from models import R2D2Network, RNDRandomNetwork

import torch.optim as optim
import torch.nn as nn


class R2D2Agent:
  def __init__(self, device, config: Config):
    self._device = device
    self._config = config

    self.online_net = R2D2Network(device, config).to(device)
    self.target_net = R2D2Network(device, config).to(device)

    self._optimizer = optim.Adam(
      params=self.online_net.parameters(),
      lr=config.adam_r2d2_learning_rate,
      betas=(config.adam_beta1, config.adam_beta2),
      eps=config.adam_epsilon)

  def set_weight(self, weight):
    self.online_net.set_weight(weight)
    self.target_net.set_weight(weight)

  def get_weight(self):
    return self.e_net.get_weight(), self.i_net.get_weight()

  def update_target(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def get_qvalues(self, model_input, input: AgentInput):
    with torch.no_grad():
      # burn in
      _, online_lstm_state = self.online_net(model_input)
      _, target_lstm_state = self.target_net(model_input)

    # 推論
    online_input = AgentInput(
      state=input.state,
      prev_action=input.prev_action,
      e_prev_reward=input.e_prev_reward,
      i_prev_reward=input.i_prev_reward,
      meta_index=input.meta_index,
      prev_lstm_state=online_lstm_state
    )

    target_input = AgentInput(
      state=input.state.clone(),
      prev_action=input.prev_action.clone(),
      e_prev_reward=input.e_prev_reward.clone(),
      i_prev_reward=input.i_prev_reward.clone(),
      meta_index=input.meta_index.clone(),
      prev_lstm_state=target_lstm_state
    )

    online_out, _ = self.online_net(online_input)
    target_out, _ = self.target_net(target_input)

    return online_out, target_out

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
