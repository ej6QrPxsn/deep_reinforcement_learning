import numpy as np
import torch
import torch.nn as nn

from utils import AgentInput, SelectActionOutput, select_actions


class DQNAgent:
  def __init__(self, device, config):

    self.qnet = QNetwork(device, config).to(device)
    self.target_qnet = QNetwork(device, config).to(device)


class QNetwork(nn.Module):

  def __init__(self, device, config, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(QNetwork, self).__init__()
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
        nn.Linear(7 * 7 * 64, 512),
        nn.ReLU()
    )

    self.lstm = nn.LSTM(input_size=512, hidden_size=config.lstm_state_size,
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

  def forward(self, agent_input: AgentInput):
    batch_size = agent_input.state.shape[0]
    seq_len = agent_input.state.shape[1]

    # batch, seq -> batch * seq
    feature_in = agent_input.state.reshape(-1, *agent_input.state.shape[2:])
    feature_out = self.feature(feature_in / 255.)

    # batch * seq -> batch, seq
    lstm_in = feature_out.reshape(batch_size, seq_len, -1)
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

  def select_actions(self, agent_input, epsilons, batch_size):
    qvalues, (hidden_state, cell_state) = self.forward(agent_input)
    actions, policies = select_actions(qvalues, self.config.action_space, epsilons, self.device, batch_size)
    qvalues = qvalues.squeeze(1).cpu().detach().numpy().copy()

    return SelectActionOutput(
      action=actions,
      qvalue=qvalues,
      policy=policies,
      # num_layer, batch -> batch, num_layer
      hidden_state=hidden_state.permute(1, 0, 2).cpu().detach().numpy().copy(),
      cell_state=cell_state.permute(1, 0, 2).cpu().detach().numpy().copy()
    )
