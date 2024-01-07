import torch
from config import Config
from model import DecisionTransformer, Input


def ready_input(config):
  reward_input = torch.rand(config.batch_size, config.seq_len, 1)
  action_input = torch.randint(
      0, config.action_size, size=(config.batch_size, config.seq_len, 1)).to(torch.float32)
  state_input = torch.rand(
      config.batch_size, config.seq_len, config.state_size)
  timstep_input = torch.randint(0, config.seq_len, size=(config.batch_size, config.seq_len, 1))
  input = Input(
      reward=reward_input,
      action=action_input,
      state=state_input,
      timestep=timstep_input
  )

  return input


def test_init():
  config = Config()

  config.batch_size = 7
  config.seq_len = 2
  config.action_size = 4
  config.state_size = 5
  config.model_dimension = 6
  config.num_head = 3

  input = ready_input(config)

  decision_transformer = DecisionTransformer(config)
  emb = decision_transformer.get_embeddings(input)
  torch.testing.assert_close(
      emb.shape, (config.batch_size, config.seq_len * 3, config.model_dimension))
