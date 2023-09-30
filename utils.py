import numpy as np
import torch
from config import Config
from data_type import AgentInput, AgentInputData, ComputeLossInput


def to_agent_input(agent_input_data: AgentInputData, device) -> AgentInput:
  return AgentInput(
      state=torch.from_numpy(agent_input_data.state.copy()).to(torch.float32).to(device),
      prev_action=torch.from_numpy(agent_input_data.prev_action.copy()).to(torch.int64).to(device),
      e_prev_reward=torch.from_numpy(agent_input_data.e_prev_reward.copy()).to(torch.float32).to(device),
      i_prev_reward=torch.from_numpy(agent_input_data.i_prev_reward.copy()).to(torch.float32).to(device),
      meta_index=torch.from_numpy(agent_input_data.meta_index.copy()).to(torch.int64).to(device),
      prev_lstm_state=(
        # batch, num_layer -> num_layer, batch
        torch.from_numpy(agent_input_data.e_lstm_states.hidden_state.copy()).permute(1, 0, 2).to(device),
        torch.from_numpy(agent_input_data.e_lstm_states.cell_state.copy()).permute(1, 0, 2).to(device)
      )
  ), AgentInput(
      state=torch.from_numpy(agent_input_data.state.copy()).to(torch.float32).to(device),
      prev_action=torch.from_numpy(agent_input_data.prev_action.copy()).to(torch.int64).to(device),
      e_prev_reward=torch.from_numpy(agent_input_data.e_prev_reward.copy()).to(torch.float32).to(device),
      i_prev_reward=torch.from_numpy(agent_input_data.i_prev_reward.copy()).to(torch.float32).to(device),
      meta_index=torch.from_numpy(agent_input_data.meta_index.copy()).to(torch.int64).to(device),
      prev_lstm_state=(
        # batch, num_layer -> num_layer, batch
        torch.from_numpy(agent_input_data.i_lstm_states.hidden_state.copy()).permute(1, 0, 2).to(device),
        torch.from_numpy(agent_input_data.i_lstm_states.cell_state.copy()).permute(1, 0, 2).to(device)
      )
  )


def get_agent_input_burn_in_from_transition(transition, device, config):
  prev_action = np.concatenate([
    transition["prev_action"][:, np.newaxis],
    transition["action"][:, :config.replay_period - 1]
  ], axis=1)
  e_prev_reward = np.concatenate([
    transition["e_prev_reward"][:, np.newaxis],
    transition["e_reward"][:, :config.replay_period - 1]
  ], axis=1)
  i_prev_reward = np.concatenate([
    transition["i_prev_reward"][:, np.newaxis],
    transition["i_reward"][:, :config.replay_period - 1]
  ], axis=1)

  meta_index = torch.empty(prev_action.shape, dtype=int, device=device)
  meta_index[:] = torch.from_numpy(transition["meta_index"][:, np.newaxis].copy())

  return AgentInput(
    state=torch.from_numpy(transition["state"][:, :config.replay_period].copy()).to(torch.float32).to(device),
    prev_action=torch.from_numpy(prev_action.copy()).to(torch.int64).to(device),
    e_prev_reward=torch.from_numpy(e_prev_reward.copy()).unsqueeze(-1).to(torch.float32).to(device),
    i_prev_reward=torch.from_numpy(i_prev_reward.copy()).unsqueeze(-1).to(torch.float32).to(device),
    meta_index=meta_index,
    prev_lstm_state=(
      # batch, num_layer -> num_layer, batch
      torch.from_numpy(transition["e_prev_hidden_state"].copy()).to(torch.float32).permute(1, 0, 2).to(device),
      torch.from_numpy(transition["e_prev_cell_state"].copy()).to(torch.float32).permute(1, 0, 2).to(device)
    )
  ), AgentInput(
    state=torch.from_numpy(transition["state"][:, :config.replay_period].copy()).to(torch.float32).to(device),
    prev_action=torch.from_numpy(prev_action.copy()).to(torch.int64).to(device),
    e_prev_reward=torch.from_numpy(e_prev_reward.copy()).unsqueeze(-1).to(torch.float32).to(device),
    i_prev_reward=torch.from_numpy(i_prev_reward.copy()).unsqueeze(-1).to(torch.float32).to(device),
    meta_index=meta_index,
    prev_lstm_state=(
      # batch, num_layer -> num_layer, batch
      torch.from_numpy(transition["i_prev_hidden_state"].copy()).to(torch.float32).permute(1, 0, 2).to(device),
      torch.from_numpy(transition["i_prev_cell_state"].copy()).to(torch.float32).permute(1, 0, 2).to(device)
    )
  )


def get_agent_input_from_transition(transition, device, config):
  meta_index = torch.empty(transition["action"][:, config.replay_period - 1:-1].shape, dtype=int, device=device)
  meta_index[:] = torch.from_numpy(transition["meta_index"][:, np.newaxis].copy())

  return AgentInput(
    state=torch.from_numpy(transition["state"][:, config.replay_period:].copy()).to(torch.float32).to(device),
    prev_action=torch.from_numpy(transition["action"][:, config.replay_period - 1:-1].copy()).to(torch.int64).to(device),
    e_prev_reward=torch.from_numpy(transition["e_reward"][:, config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(device),
    i_prev_reward=torch.from_numpy(transition["i_reward"][:, config.replay_period - 1:-1].copy()).unsqueeze(-1).to(torch.float32).to(device),
    meta_index=meta_index,
    prev_lstm_state=None
  )


def get_loss_input_for_replay(transitions, config: Config, device, beta_table, gamma_table) -> ComputeLossInput:
  beta = beta_table[transitions["meta_index"]][:, np.newaxis]
  gamma = gamma_table[transitions["meta_index"]]
  rewards = transitions["e_reward"][:, config.replay_period:] + beta * transitions["i_reward"][:, config.replay_period:]

  target_epsilon = torch.empty(transitions["action"][:, config.replay_period:].shape, dtype=torch.float32, device=device)
  target_epsilon[:] = torch.from_numpy(beta.copy())

  return ComputeLossInput(
      action=torch.from_numpy(transitions["action"][:, config.replay_period:].copy()).to(torch.int64).unsqueeze(-1).to(device),
      reward=torch.from_numpy(rewards.copy()).to(torch.float32).to(device),
      done=torch.from_numpy(transitions["done"][:, config.replay_period:].copy()).to(torch.bool).to(device),
      policy=torch.from_numpy(transitions["policy"][:, config.replay_period:].copy()).unsqueeze(-1).to(torch.float32).to(device),
      beta=target_epsilon,
      gamma=torch.from_numpy(gamma[:, np.newaxis].copy()).to(torch.float32).to(device),
  )


def get_loss_input_for_train(transitions, config: Config, device, beta_table, gamma_table) -> ComputeLossInput:
  beta = beta_table[transitions["meta_index"]][:, np.newaxis]
  gamma = gamma_table[transitions["meta_index"]]

  target_epsilon = torch.empty(transitions["action"][:, config.replay_period:].shape, dtype=torch.float32, device=device)
  target_epsilon[:] = torch.from_numpy(beta.copy())

  return ComputeLossInput(
      action=torch.from_numpy(transitions["action"][:, config.replay_period:].copy()).to(torch.int64).unsqueeze(-1).to(device),
      reward=torch.from_numpy(transitions["e_reward"][:, config.replay_period:].copy()).to(torch.float32).to(device),
      done=torch.from_numpy(transitions["done"][:, config.replay_period:].copy()).to(torch.bool).to(device),
      policy=torch.from_numpy(transitions["policy"][:, config.replay_period:].copy()).unsqueeze(-1).to(torch.float32).to(device),
      beta=target_epsilon,
      gamma=torch.from_numpy(gamma[:, np.newaxis].copy()).to(torch.float32).to(device),
  ), ComputeLossInput(
      action=torch.from_numpy(transitions["action"][:, config.replay_period:].copy()).to(torch.int64).unsqueeze(-1).to(device),
      reward=torch.from_numpy(transitions["i_reward"][:, config.replay_period:].copy()).to(torch.float32).to(device),
      done=torch.from_numpy(transitions["done"][:, config.replay_period:].copy()).to(torch.bool).to(device),
      policy=torch.from_numpy(transitions["policy"][:, config.replay_period:].copy()).unsqueeze(-1).to(torch.float32).to(device),
      beta=target_epsilon,
      gamma=torch.from_numpy(gamma[:, np.newaxis].copy()).to(torch.float32).to(device),
  )


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def get_beta_table(config):
  table = np.empty(config.num_arms)
  for i in range(config.num_arms):
    if i == 0:
      table[i] = 0
    elif i == config.num_arms - 1:
      table[i] = config.intrinsic_reward_scale
    else:
      table[i] = config.intrinsic_reward_scale * sigmoid(10 * (2 * i - (config.num_arms - 2)) / (config.num_arms - 2))

  return table


def get_gamma_table(config):
  table = np.empty(config.num_arms)
  for i in range(config.num_arms):
    if i == 0:
      table[i] = config.gamma_0
    elif i < 7:
      table[i] = config.gamma_1 + (config.gamma_0 - config.gamma_1) * sigmoid(10 * (2 * i - 6) / 6)
    elif i == 7:
      table[i] = config.gamma_1
    else:
      value1 = (config.num_arms - 9) * np.log(1 - config.gamma_1)
      value2 = (i - 8) * np.log(1 - config.gamma_2)
      table[i] = 1 - np.exp((value1 + value2) / (config.num_arms - 9))

  return table
