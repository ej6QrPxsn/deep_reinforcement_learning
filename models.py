import numpy as np
import torch
import torch.nn as nn
from config import Config

from utils import AgentInput
import torch.nn.functional as F


class R2D2Network(nn.Module):

  def __init__(self, device, config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(R2D2Network, self).__init__()
    self.config: Config = config
    self.device = device
    self.feature = nn.Sequential(
        # (in - (kernel - 1) - 1) / stride + 1
        # (84 - 8) / 4 + 1 = 20
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        # (20 - 4) / 2 + 1 = 9
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        # (9 - 3) / 1 + 1 = 7
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 512),
        nn.ReLU()
    )

    self.lstm = nn.LSTM(input_size=512 + config.action_space + config.num_arms + 2,
                        hidden_size=config.lstm_state_size,
                        num_layers=config.lstm_num_layers, batch_first=True)

    self.value = nn.Sequential(
        nn.Linear(config.lstm_state_size, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )

    self.advantage = nn.Sequential(
        nn.Linear(config.lstm_state_size, 512),
        nn.ReLU(),
        nn.Linear(512, config.action_space),
    )

  def set_weight(self, weight):
    self.load_state_dict(weight)

  def get_weight(self):
    return self.state_dict()

  def forward(self, agent_input: AgentInput):
    batch_size = agent_input.state.shape[0]
    seq_len = agent_input.state.shape[1]

    # batch, seq -> batch * seq
    feature_in = agent_input.state.reshape(-1, *agent_input.state.shape[2:])
    feature_out = self.feature(feature_in / 255.)

    prev_action_one_hot = F.one_hot(agent_input.prev_action, num_classes=self.config.action_space)
    beta_one_hot = F.one_hot(agent_input.arm_index, num_classes=self.config.num_arms)

    # batch, (burn_in + )seq, conv outputs + reward + actions
    lstm_in = torch.cat((
      # batch * seq -> batch, seq
      feature_out.reshape(batch_size, seq_len, -1),
      prev_action_one_hot,
      agent_input.prev_extrinsic_reward,
      agent_input.prev_intrinsic_reward,
      beta_one_hot,
    ), 2)

    lstm_out, lstm_states = self.lstm(lstm_in, agent_input.prev_lstm_state)

    # batch, seq -> batch * seq
    dueling_in = lstm_out.reshape(-1, *lstm_out.shape[2:])

    value_out = self.value(dueling_in)
    adv_out = self.advantage(dueling_in)

    dueling_out = value_out + adv_out - torch.mean(adv_out, -1, True)
    # batch　* seq -> batch, seq
    return dueling_out.reshape(batch_size, seq_len, *dueling_out.shape[1:]), lstm_states

  def initial_state(self, batch_size):
    return (np.zeros((batch_size, self.config.lstm_num_layers, self.config.lstm_state_size), dtype=np.float32),
            np.zeros((batch_size, self.config.lstm_num_layers, self.config.lstm_state_size), dtype=np.float32))


class RNDPredictionNetwork(nn.Module):

  def __init__(self, device, config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(RNDPredictionNetwork, self).__init__()
    self.config = config
    self.device = device
    self.feature = nn.Sequential(
        # (in - (kernel - 1) - 1) / stride + 1
        # (84 - 8) / 4 + 1 = 20
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        # (20 - 4) / 2 + 1 = 9
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        # (9 - 3) / 1 + 1 = 7
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 128),
    )

  def set_weight(self, weight):
    self.feature.load_state_dict(weight)

  def get_weight(self):
    return self.feature.state_dict()

  def forward(self, x):
    return self.feature(x / 255.)


class RNDRandomNetwork(nn.Module):

  def __init__(self, device, config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(RNDRandomNetwork, self).__init__()
    self.config = config
    self.device = device
    self.feature = nn.Sequential(
        # (in - (kernel - 1) - 1) / stride + 1
        # (84 - 8) / 4 + 1 = 20
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        # (20 - 4) / 2 + 1 = 9
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        # (9 - 3) / 1 + 1 = 7
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 128),
    )

  def forward(self, x):
    return self.feature(x / 255.)


class EmbeddingNetwork(nn.Module):

  def __init__(self, device, config: Config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(EmbeddingNetwork, self).__init__()
    self.config = config
    self.device = device
    self.feature = nn.Sequential(
        # (in - (kernel - 1) - 1) / stride + 1
        # (84 - 8) / 4 + 1 = 20
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        # (20 - 4) / 2 + 1 = 9
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        # (9 - 3) / 1 + 1 = 7
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, config.controllable_state_size),
        nn.ReLU()
    )

  def set_weight(self, weight):
    self.feature.load_state_dict(weight)

  def get_weight(self):
    return self.feature.state_dict()

  def forward(self, x):
    return self.feature(x / 255.)


class ActionPredictionNetwork(nn.Module):

  def __init__(self, device, config: Config, embedding_net):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(ActionPredictionNetwork, self).__init__()
    self.config = config
    self.device = device
    self.embedding = embedding_net

    self.siamese = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, config.action_space),
        nn.Softmax(dim=1)
    )
    self.siamese.to(device)

  def forward(self, transition):
    state = torch.from_numpy(transition["state"][:, -self.config.embedding_train_period - 1:-1].copy()).to(torch.float32).to(self.device)
    next_state = torch.from_numpy(transition["state"][:, -self.config.embedding_train_period:].copy()).to(torch.float32).to(self.device)

    ret1 = self.embedding(state.reshape(-1, *state.shape[2:]))
    ret2 = self.embedding(next_state.reshape(-1, *next_state.shape[2:]))

    return self.siamese(torch.cat([ret1, ret2], dim=1))
