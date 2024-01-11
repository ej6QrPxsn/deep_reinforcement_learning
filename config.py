

class Config:
  env_name = "BreakoutDeterministic-v4"

  # 環境実行プロセス数
  n_envs = 1
  context_length = 90

  # n_steps = 2 * 500000 * context_length // n_envs
  n_steps = 100 * context_length

  batch_size = 3
  action_size = 3
  input_type = "image"

  in_feature = 32
  # seq_len * (rtg, state, action)
  n_features = context_length * 3
  embed_dim = 128
  n_head = 8
  n_block = 6
  ffn_dim = 2048
  state_size = 84 * 84 * 4

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

  # 最適化
  adam_lr = 6 * 10 ** -4
  adam_beta = (0.9, 0.95)
