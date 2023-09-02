
import numpy as np


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
        ("beta", "f4"),
        ("gamma", "f4"),
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
      ("beta", "f4"),
      ("gamma", "f4"),
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
      ("beta", "f4"),
      ("gamma", "f4"),
      ("action", "u1"),
    ])

    self.agent_input_dtype = np.dtype([
        ("state", "u1", (1, *config.state_shape)),
        ("prev_action", "u1", (1,)),
        ("prev_extrinsic_reward", "f4", (1, 1)),
        ("prev_intrinsic_reward", "f4", (1, 1)),
        ("beta", "f4", (1, 1)),
        ("prev_hidden_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
        ("prev_cell_state", "f4", (config.lstm_num_layers, config.lstm_state_size)),
    ])
