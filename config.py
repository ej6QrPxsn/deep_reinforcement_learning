from dataclasses import dataclass


@dataclass
class Config:
  # 環境
  env_name = "BreakoutDeterministic-v4"
  shared_env_name = "shared_env_name-v4"

  action_space = None
  state_shape = None

  def init(self, action_space, state_shape):
    self.action_space = action_space
    self.state_shape = state_shape

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
  gamma_min = 0.99
  gamma_max = 0.997

  # リプレイデータ
  replay_buffer_size = 500000
  replay_buffer_min_size = 6500
  replay_buffer_add_print_size = 500
  max_trainsition_queue_size = 8

  # 損失
  adam_epsilon = 1e-3
  eval_epsilon = 0.01
  target_epsilon = 0.01
  retrace_lambda = 0.95
  target_update_period = 2500
  rescaling_epsilon = 1e-3

  # LSTM
  lstm_state_size = 512
  lstm_num_layers = 1

  # intrinsic reward
  episodic_memory_capacity = 300000
  controllable_state_size = 32
  num_kernel = 10
  kernel_cluster_distance = 0.008
  kernel_epsilon = 0.008
  kernel_pseudo_counts_constant = 0.001
  kernel_maximum_similarity = 8
  embedding_train_period = 5
  intrinsic_reward_scale = 0.3

  RND_clipping_factor = 5
  RND_train_period = 5

  num_train_log = 4
