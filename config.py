

class Config:
  # env_name = "PitfallDeterministic-v4"
  env_name = "BreakoutDeterministic-v4"
  checkpoint_path = "checkpoint.pth"

  min_replay_size = 50000
  replay_size = 500000

  target_type = "atari"

  # 書き込みプロセス数
  n_writers = 3

  # データ読み込みプロセス数
  n_loads = 3

  # 環境実行プロセス数
  n_envs = 3

  # 訓練データの割合
  train_date_ratio = 1

  # 1アクターで実行する環境数
  n_env_batches = 2

  n_val_episode = 1000
  batch_size = 128
  action_size = 3
  input_type = "image"

  context_length = 30
  block_size = context_length * 3

  # 最大ステップ数
  max_timestep = 2000

  n_steps = 2 * 500000 * context_length // (n_envs * n_env_batches)
  train_steps = (2 * 500000 - context_length) // batch_size

  in_feature = 32
  # seq_len * (rtg, state, action)
  n_features = block_size
  embed_dim = 128
  n_head = 8
  n_block = 6
  ffn_dim = 2048
  state_size = 4 * 84 * 84

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
  weight_decay = 0.1
