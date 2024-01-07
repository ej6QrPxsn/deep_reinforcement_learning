

class Config:
  seq_len = 160
  batch_size = 64
  action_size = 3

  in_feature = 512
  model_dimension = 512
  num_head = 8
  num_stack_block = 12
  ffn_dimension = 2048
  state_size = 84 * 84 * 3 * 160

