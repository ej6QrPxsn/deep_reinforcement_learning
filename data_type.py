
from typing import NamedTuple, Tuple
import numpy as np
import torch


class LstmStates(NamedTuple):
  hidden_state: np.ndarray
  cell_state: np.ndarray


class AgentInputData(NamedTuple):
  state: np.ndarray
  prev_action: np.ndarray
  e_prev_reward: np.ndarray
  i_prev_reward: np.ndarray
  meta_index: np.ndarray
  e_lstm_states: LstmStates
  i_lstm_states: LstmStates


class SelectActionOutput(NamedTuple):
  action: np.ndarray
  policy: np.ndarray
  qvalue: np.ndarray
  e_qvalue: np.ndarray
  e_lstm_states: LstmStates
  i_qvalue: np.ndarray
  i_lstm_states: LstmStates


class AgentInput(NamedTuple):
  state: torch.Tensor
  prev_action: torch.Tensor
  e_prev_reward: torch.Tensor
  i_prev_reward: torch.Tensor
  meta_index: torch.Tensor
  prev_lstm_state: Tuple[torch.Tensor, torch.Tensor]


class AgentOutput(NamedTuple):
  qvalue: torch.Tensor
  hidden_state: torch.Tensor
  cell_state: torch.Tensor


class ComputeLossInput(NamedTuple):
  action: torch.Tensor
  reward: torch.Tensor
  done: torch.Tensor
  policy: torch.Tensor
  beta: torch.Tensor
  gamma: torch.Tensor


class DataType():
  def __init__(self, config):
    # シーケンス蓄積用遷移データ
    self.work_transition_dtype = np.dtype([
        ("state", "u1", config.state_shape),
        ("action", "u1"),
        ("e_reward", "f4"),
        ("i_reward", "f4"),
        ("done", "?"),
        ("policy", "f4"),
        ("qvalue", "f4", config.action_space),
        ("meta_index", "u1"),
        ("prev_action", "u1"),
        ("e_prev_reward", "f4"),
        ("i_prev_reward", "f4"),
        ("e_qvalue", "f4", config.action_space),
        ("e_prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("e_prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("i_qvalue", "f4", config.action_space),
        ("i_prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("i_prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
    ])

    # 遷移データ
    self.transition_dtype = np.dtype([
      ("state", "u1", (config.seq_len + 1, *config.state_shape)),
      ("action", "u1", config.seq_len + 1),
      ("e_reward", "f4", config.seq_len + 1),
      ("i_reward", "f4", config.seq_len + 1),
      ("done", "?", config.seq_len + 1),
      ("policy", "f4", config.seq_len + 1),
      ("meta_index", "u1"),
      ("prev_action", "u1"),
      ("e_prev_reward", "f4"),
      ("i_prev_reward", "f4"),
      ("e_prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
      ("e_prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
      ("i_prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
      ("i_prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
    ])

    # 環境データ
    self.env_dtype = np.dtype([
      ("next_state", "u1", config.state_shape),
      ("reward", "f4"),
      ("done", "?"),
      ("meta_index", "u1"),
      ("action", "u1"),
    ])

    self.agent_input_dtype = np.dtype([
        ("state", "u1", (1, *config.state_shape)),
        ("prev_action", "u1", (1,)),
        ("e_prev_reward", "f4", (1, 1)),
        ("i_prev_reward", "f4", (1, 1)),
        ("meta_index", "u1", (1,)),
        ("e_prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("e_prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("i_prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("i_prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
    ])
