import torch
from config import Config
from model import DecisionTransformer, Input, MultiHeadLayer


def ready_input(config: Config):
  reward_input = torch.rand(config.batch_size, config.context_length, 1)
  action_input = torch.randint(
      0, config.action_size, size=(config.batch_size, config.context_length, 1)).to(torch.float32)
  state_input = torch.rand(
      config.batch_size, config.context_length, config.state_size)
  timstep_input = torch.randint(0, config.context_length, size=(config.batch_size, config.context_length, 1))
  input = Input(
      rtg=reward_input,
      action=action_input,
      state=state_input,
      timestep=timstep_input
  )

  return input


def test_init():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = Config()

  config.input_type = "text"
  config.batch_size = 7
  config.context_length = 2
  config.action_size = 4
  config.state_size = 5
  config.embed_dim = 6
  config.n_head = 3

  input = ready_input(config)

  decision_transformer = DecisionTransformer(config, device)
  emb = decision_transformer.get_embeddings(input)
  torch.testing.assert_close(
      emb.shape, (config.batch_size, config.context_length * 3, config.embed_dim))


def test_MultiHeadLayer():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = Config()

  config.input_type = "text"
  config.batch_size = 7
  config.context_length = 2
  config.action_size = 4
  config.state_size = 5
  config.embed_dim = 6
  config.n_head = 3

  input = ready_input(config)

  decision_transformer = DecisionTransformer(config, device)
  emb = decision_transformer.get_embeddings(input)
  multi_head_layer = MultiHeadLayer(config)
  out = multi_head_layer(emb)
  torch.testing.assert_close(
      out.shape, (config.batch_size, config.context_length * 3, config.embed_dim))


def test_DecisionTransformer():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = Config()

  config.input_type = "text"
  config.batch_size = 7
  config.context_length = 2
  config.action_size = 4
  config.state_size = 5
  config.embed_dim = 6
  config.n_head = 3

  input = ready_input(config)

  decision_transformer = DecisionTransformer(config, device)
  out = decision_transformer(input)
  torch.testing.assert_close(
      out.shape, (config.batch_size, config.action_size))
