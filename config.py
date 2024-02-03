

class Config:
  # env_name = "PitfallDeterministic-v4"
  env_name = "BreakoutDeterministic-v4"
  checkpoint_path = "checkpoint.pth"
  n_workers = 2

  min_replay_size = 25000
  replay_size = 500000

  # データ読み込みプロセス数
  n_loads = 10

  # 訓練データの割合
  train_date_ratio = 1

  n_val_episode = 1000
  batch_size = 32
  action_size = 3
  input_type = "image"

  # 環境実行プロセス数
  n_envs = 10
  context_length = 90
  max_timestep = 100000

  n_steps = 2 * 500000 * context_length // n_envs
  train_steps = (2 * 500000 - context_length) // batch_size

  in_feature = 32
  # seq_len * (rtg, state, action)
  n_features = context_length * 3
  embed_dim = 128
  n_head = 8
  n_block = 6
  ffn_dim = 2048
  state_size = 4 * 84 * 84

  # 1アクターで実行する環境数
  n_env_batches = 1
  state_shape = (4, 84, 84)

  # 訓練データ作成
  train_data_dir = "data/train"
  validate_data_dir = "data/validate"
  train_filename = "train-%05d.tar"
  validate_filename = "validate-%05d.tar"
  shard_size = 50 * 1000 * 1000
  data_queue_max_size = 1000
  max_epochs = 5
  embed_drop = 0.1

  # 最適化
  adam_lr = 6 * 10 ** -4
  adam_beta = (0.9, 0.95)
  grad_norm_clip = 1.0
  use_amp = False
