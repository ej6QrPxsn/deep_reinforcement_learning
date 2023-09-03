import numpy as np
import torch
from config import Config
from models import ActionPredictionNetwork, R2D2Network, RNDRandomNetwork

from utils import AgentInput
import torch.optim as optim


class R2D2Agent:
  def __init__(self, device, config: Config):
    self.device = device
    self.config = config

    self.online_net = R2D2Network(self.device, self.config).to(self.device)
    self.target_net = R2D2Network(self.device, self.config).to(self.device)
    self._optimizer = optim.Adam(self.online_net.parameters(),
                                 lr=config.r2d2_learning_rate,
                                 betas=config.adam_betas,
                                 eps=config.adam_epsilon)

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

    arm_index = torch.empty(prev_action.shape, dtype=int, device=self.device)
    arm_index[:] = torch.from_numpy(transition["arm_index"][:, np.newaxis].copy())

    return AgentInput(
      state=torch.from_numpy(transition["state"][:, :self.config.replay_period].copy()).to(torch.float32).to(self.device),
      prev_action=torch.from_numpy(prev_action.copy()).to(torch.int64).to(self.device),
      prev_extrinsic_reward=torch.from_numpy(prev_extrinsic_reward.copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      prev_intrinsic_reward=torch.from_numpy(prev_intrinsic_reward.copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      arm_index=arm_index,
      prev_lstm_state=(
        # batch, num_layer -> num_layer, batch
        torch.from_numpy(transition["prev_hidden_state"].copy()).to(torch.float32).permute(1, 0, 2).to(self.device),
        torch.from_numpy(transition["prev_cell_state"].copy()).to(torch.float32).permute(1, 0, 2).to(self.device)
      )
    )

  def get_agent_input_from_transition(self, transition, lstm_state):
    arm_index = torch.empty(transition["action"][:, self.config.replay_period - 1:-1].shape, dtype=int, device=self.device)
    arm_index[:] = torch.from_numpy(transition["arm_index"][:, np.newaxis].copy())

    return AgentInput(
      state=torch.from_numpy(transition["state"][:, self.config.replay_period:].copy()).to(torch.float32).to(self.device),
      prev_action=torch.from_numpy(transition["action"][:, self.config.replay_period - 1:-1].copy()).to(torch.int64).to(self.device),
      prev_extrinsic_reward=torch.from_numpy(transition["extrinsic_reward"][:, self.config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      prev_intrinsic_reward=torch.from_numpy(transition["intrinsic_reward"][:, self.config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(self.device),
      arm_index=arm_index,
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

    # 勾配クリップ
    # torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.adam_clip_norm)
    # 勾配反映
    self._optimizer.step()


class RNDAgent:

  def __init__(self, device, config, predict, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    self.config = config
    self.device = device

    self.predict = predict

    self.random = RNDRandomNetwork(device, config)
    self.random.to(device)

    self.random.load_state_dict(self.predict.state_dict())

    self.criterion = torch.nn.MSELoss(reduction="none")
    self.optimizer = torch.optim.Adam(self.predict.parameters(),
                                      lr=config.rnd_learning_rate,
                                      betas=config.adam_betas,
                                      eps=config.adam_epsilon)

  def get_loss(self, rand_out, predict_out):
    loss = self.criterion(rand_out, predict_out)
    return loss.mean(1)

  def train(self, transition):
    state = torch.from_numpy(transition["state"][:, -self.config.embedding_train_period:].copy()).to(torch.float32).to(self.device)

    rand_out = self.random(state.reshape(-1, *state.shape[2:]))
    predict_out = self.predict(state.reshape(-1, *state.shape[2:]))

    loss = self.criterion(rand_out, predict_out)

    self.optimizer.zero_grad()
    loss.mean().backward()
    # 勾配クリップ
    # torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.adam_clip_norm)
    self.optimizer.step()


class ActionPredictionAgent:
  def __init__(self, device, config, embedding_net):
    self.device = device
    self.config = config

    self.action_predict_net = ActionPredictionNetwork(self.device, self.config, embedding_net).to(self.device)

    self.criterion = torch.nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.action_predict_net.parameters(),
                                      lr=config.action_prediction_learning_rate,
                                      betas=config.adam_betas,
                                      eps=config.adam_epsilon)

  def train(self, transition):
    output = self.action_predict_net(transition)

    action = torch.from_numpy(transition["action"][:, -self.config.embedding_train_period - 1:-1].copy()).to(torch.int64).to(self.device)
    loss = self.criterion(output, action.reshape(-1, *action.shape[2:]))

    self.optimizer.zero_grad()
    loss.backward()
    # 勾配クリップ
    # torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.adam_clip_norm)
    self.optimizer.step()
