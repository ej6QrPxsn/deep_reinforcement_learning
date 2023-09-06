
from typing import NamedTuple, Tuple
import numpy as np
import torch


class AgentInputData(NamedTuple):
  state: np.ndarray
  prev_action: np.ndarray
  prev_extrinsic_reward: np.ndarray
  prev_intrinsic_reward: np.ndarray
  policy_index: np.ndarray
  hidden_state: np.ndarray
  cell_state: np.ndarray


class SelectActionOutput(NamedTuple):
  action: np.ndarray
  qvalue: np.ndarray
  policy: np.ndarray
  hidden_state: np.ndarray
  cell_state: np.ndarray


class AgentInput(NamedTuple):
  state: torch.Tensor
  prev_action: torch.Tensor
  prev_extrinsic_reward: torch.Tensor
  prev_intrinsic_reward: torch.Tensor
  policy_index: torch.Tensor
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
        ("extrinsic_reward", "f4"),
        ("intrinsic_reward", "f4"),
        ("done", "?"),
        ("policy", "f4"),
        ("qvalue", "f4", config.action_space),
        ("policy_index", "u1"),
        ("prev_action", "u1"),
        ("prev_extrinsic_reward", "f4"),
        ("prev_intrinsic_reward", "f4"),
        ("prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
    ])

    # 遷移データ
    self.transition_dtype = np.dtype([
      ("state", "u1", (config.seq_len + 1, *config.state_shape)),
      ("action", "u1", config.seq_len + 1),
      ("extrinsic_reward", "f4", config.seq_len + 1),
      ("intrinsic_reward", "f4", config.seq_len + 1),
      ("done", "?", config.seq_len + 1),
      ("policy", "f4", config.seq_len + 1),
      ("policy_index", "u1"),
      ("prev_action", "u1"),
      ("prev_extrinsic_reward", "f4"),
      ("prev_intrinsic_reward", "f4"),
      ("prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
      ("prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
    ])

    # 環境データ
    self.env_dtype = np.dtype([
      ("next_state", "u1", config.state_shape),
      ("reward", "f4"),
      ("done", "?"),
      ("policy_index", "u1"),
      ("action", "u1"),
    ])

    self.agent_input_dtype = np.dtype([
        ("state", "u1", (1, *config.state_shape)),
        ("prev_action", "u1", (1,)),
        ("prev_extrinsic_reward", "f4", (1, 1)),
        ("prev_intrinsic_reward", "f4", (1, 1)),
        ("policy_index", "u1", (1,)),
        ("prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
    ])
