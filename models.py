import numpy as np
import torch
import torch.nn as nn
from config import Config
from data_type import LstmStates, SelectActionOutput

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
    self._config = config
    self._device = device
    self._feature = nn.Sequential(
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

    self.lstm = nn.LSTM(input_size=512 + config.action_space + config.num_arms + 2, hidden_size=config.lstm_state_size,
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
    feature_out = self._feature(feature_in / 255.)

    prev_action_one_hot = F.one_hot(agent_input.prev_action, num_classes=self._config.action_space)
    policy_one_hot = F.one_hot(agent_input.meta_index, num_classes=self._config.num_arms)

    # batch, (burn_in + )seq, conv outputs + reward + actions
    lstm_in = torch.cat((
      # batch * seq -> batch, seq
      feature_out.reshape(batch_size, seq_len, -1),
      prev_action_one_hot,
      agent_input.e_prev_reward,
      agent_input.i_prev_reward,
      policy_one_hot,
    ), 2)

    lstm_out, lstm_states = self.lstm(lstm_in, agent_input.prev_lstm_state)

    # batch, seq -> batch * seq
    dueling_in = lstm_out.reshape(-1, *lstm_out.shape[2:])

    value_out = self.value(dueling_in)
    adv_out = self.advantage(dueling_in)

    dueling_out = value_out + adv_out - torch.mean(adv_out, -1, True)
    # batch　* seq -> batch, seq
    return dueling_out.reshape(batch_size, seq_len, *dueling_out.shape[1:]), lstm_states

  def get_qvalue(self, burn_in_input, input: AgentInput):
    with torch.no_grad():
      # burn in
      _, lstm_state = self.forward(burn_in_input)

    # 推論
    output, _ = self.forward(
      AgentInput(
          input.state.clone(),
          input.prev_action,
          input.e_prev_reward.clone(),
          input.i_prev_reward.clone(),
          input.meta_index,
          lstm_state
      )
    )

    return output


class Agent57Network():

  def __init__(self, device, config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    self._config = config
    self._device = device
    self.e_net = R2D2Network(device, config).to(device)
    self.i_net = R2D2Network(device, config).to(device)

  def share_memory(self):
    self.e_net.share_memory()
    self.i_net.share_memory()

  def initial_state(self, batch_size):
    return LstmStates(
      np.zeros((batch_size, self._config.lstm_num_layers, self._config.lstm_state_size), dtype=np.float32),
      np.zeros((batch_size, self._config.lstm_num_layers, self._config.lstm_state_size), dtype=np.float32)
    ), LstmStates(
      np.zeros((batch_size, self._config.lstm_num_layers, self._config.lstm_state_size), dtype=np.float32),
      np.zeros((batch_size, self._config.lstm_num_layers, self._config.lstm_state_size), dtype=np.float32)
    )

  def set_weight(self, e_weight, i_weight):
    self.e_net.set_weight(e_weight)
    self.i_net.set_weight(i_weight)

  def get_weight(self):
    return self.e_net.get_weight(), self.i_net.get_weight()

  def select_actions(self, e_qvalues, e_lstm_states, i_qvalues, i_lstm_states, epsilons, beta, batch_size, action_getter):
    e_hidden_state, e_cell_state = e_lstm_states
    i_hidden_state, i_cell_state = i_lstm_states

    qvalues = e_qvalues + beta.unsqueeze(-1) * i_qvalues
    actions, policies = action_getter.select_actions(qvalues, epsilons, batch_size)
    qvalues = qvalues.squeeze(1).cpu().detach().numpy().copy(),

    return SelectActionOutput(
      action=actions,
      policy=policies,
      qvalue=qvalues,
      e_qvalue=e_qvalues.squeeze(1).cpu().detach().numpy().copy(),
      e_lstm_states=LstmStates(
        hidden_state=e_hidden_state.permute(1, 0, 2).cpu().detach().numpy().copy(),
        cell_state=e_cell_state.permute(1, 0, 2).cpu().detach().numpy().copy(),
      ),
      i_qvalue=i_qvalues.squeeze(1).cpu().detach().numpy().copy(),
      i_lstm_states=LstmStates(
        hidden_state=i_hidden_state.permute(1, 0, 2).cpu().detach().numpy().copy(),
        cell_state=i_cell_state.permute(1, 0, 2).cpu().detach().numpy().copy(),
      )
    )


class RNDPredictionNetwork(nn.Module):

  def __init__(self, device, config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(RNDPredictionNetwork, self).__init__()
    self._config = config
    self._device = device
    self._feature = nn.Sequential(
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
    self._feature.share_memory()
    self._feature.to(device)

  def forward(self, x):
    return self._feature(x / 255.)


class RNDRandomNetwork(nn.Module):

  def __init__(self, device, config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(RNDRandomNetwork, self).__init__()
    self._config = config
    self._device = device
    self._feature = nn.Sequential(
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
    return self._feature(x / 255.)


class EmbeddingNetwork(nn.Module):

  def __init__(self, device, config: Config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(EmbeddingNetwork, self).__init__()
    self._config = config
    self._device = device
    self._feature = nn.Sequential(
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
    self._feature.share_memory()
    self._feature.to(device)

  def forward(self, x):
    return self._feature(x / 255.)
