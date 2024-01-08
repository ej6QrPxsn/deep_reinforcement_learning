

class Config:
  env_name = "BreakoutDeterministic-v4"

  # 環境実行プロセス数
  n_envs = 1
  context_length = 90

  n_steps = 2 * 500000 // n_envs

  batch_size = 128
  action_size = 3
  input_type = "image"

  in_feature = 32
  # seq_len * (rtg, state, action)
  n_features = context_length * 3
  embed_dim = 128
  n_head = 8
  n_block = 6
  ffn_dim = 2048
  state_size = 84 * 84 * 3

  # 1アクターで実行する環境数
  n_env_batches = 1
  state_shape = (4, 84, 84)

  # 訓練データ作成
  data_dir = "data"
  train_filename = f"{data_dir}/train-%05d.tar"
  validate_filename = f"{data_dir}/validate-%05d.tar"
  shard_size = 5. * 1000 * 1000
  data_queue_max_size = 1000
