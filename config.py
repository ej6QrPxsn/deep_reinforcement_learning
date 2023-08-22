from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
  # 環境
  env_name = "BreakoutDeterministic-v4"
  shared_env_name = "shared_env_name-v4"

  action_space = None
  state_shape = None
  transition_dtype = None

  def init(self, action_space, state_shape):
    self.action_space = action_space
    self.state_shape = state_shape

    # 遷移データ定義
    self.transition_dtype = np.dtype([
      ('state', 'u1', (self.seq_len + 1, *self.state_shape)),
      ('action', 'u1', self.seq_len + 1),
      ('reward', 'f4', self.seq_len + 1),
      ('done', '?', self.seq_len + 1),
      ('policy', 'f4', self.seq_len + 1),
      ('beta', 'f4', self.seq_len + 1),
      ('gamma', 'f4', self.seq_len + 1),
      ('prev_action', 'u1'),
      ('prev_reward', 'f4'),
      ('prev_hidden_state', 'f4', (self.lstm_num_layers, self.lstm_state_size)),
      ('prev_cell_state', 'f4', (self.lstm_num_layers, self.lstm_state_size)),
    ])

    # 環境データ定義
    self.env_dtype = np.dtype([
      ('next_state', 'u1', state_shape),
      ('reward', 'f4'),
      ('done', '?'),
      ('beta', 'f4'),
      ('gamma', 'f4'),
      ('action', 'u1'),
    ])

  # データ共有
  shared_env_name = "shared_env_name"
  shared_transition_name = "shared_transition_name"

  # バッチ数
  batch_size = 64

  replay_period = 32
  trace_length = 32

  # シーケンス数
  seq_len = replay_period + trace_length

  # アクター数
  num_actors = 16
  # 1アクターあたり環境数
  num_env_batches = 16
  # 全環境数
  num_train_envs = num_actors * num_env_batches
  # 推論プロセス数
  num_inferences = 4

  # 評価環境
  num_eval_envs = 1

  # sliding window UCB
  num_arms = 32
  bandit_window_size = 90
  bandit_UCB_beta = 1
  bandit_epsilon = 0.5
  epsilon_beta = 1 - 1e-6
  gamma_min = 0.99
  gamma_max = 0.997

  # リプレイデータ
  replay_buffer_size = 500000
  replay_buffer_min_size = 6500
  replay_buffer_add_print_size = 500
  max_trainsition_queue_size = 8

  # 損失
  eval_epsilon = 1e-3
  epsilon = 1e-3
  retrace_lambda = 0.95
  target_update_period = 2500
  rescaling_epsilon = 1e-3

  # LSTM
  lstm_state_size = 512
  lstm_num_layers = 1

  num_train_log = 4
