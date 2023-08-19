import torch
import torch.nn as nn

from utils import AgentInput, SelectActionOutput, select_actions


class DQNAgent:
  def __init__(self, device, action_space):

    self.qnet = QNetwork(device, action_space).to(device)
    self.target_qnet = QNetwork(device, action_space).to(device)


class QNetwork(nn.Module):

  def __init__(self, device, n_actions=14, in_channels=4):
    """
    Initialize Deep Q Network

    Args:
        in_channels (int): number of input channels
        n_actions (int): number of outputs
    """
    super(QNetwork, self).__init__()
    self.action_space = n_actions
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

    self.value = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )

    self.advantage = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_actions),
    )

  def forward(self, agent_input: AgentInput):
    batch_size = agent_input.state.shape[0]
    seq_size = agent_input.state.shape[1]

    # batch, seq -> batch * seq
    feature_in = agent_input.state.reshape(-1, *agent_input.state.shape[2:])
    feature_out = self.feature(feature_in / 255.)
    value_out = self.value(feature_out)
    adv_out = self.advantage(feature_out)

    dueling_out = value_out + adv_out - torch.mean(adv_out, -1, True)
    return dueling_out.reshape(batch_size, seq_size, *dueling_out.shape[1:])

  def select_actions(self, agent_input, epsilons, batch_size):
    qvalues = self.forward(agent_input)
    actions, policies = select_actions(qvalues, self.action_space, epsilons, self.device, batch_size)
    qvalues = qvalues.squeeze(1).cpu().detach().numpy().copy()

    return SelectActionOutput(
      action=actions,
      qvalue=qvalues,
      policy=policies,
    )
