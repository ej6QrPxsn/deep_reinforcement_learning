from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class InputWeights(NamedTuple):
  query: nn.Linear
  key: nn.Linear
  value: nn.Linear


class Input(NamedTuple):
  rtg: torch.Tensor
  state: torch.Tensor
  action: torch.Tensor
  timestep: int


class DecisionTransformer(nn.Module):
  def __init__(self, config: Config) -> None:
    super(DecisionTransformer, self).__init__()

    self.config = config

    if config.input_type == "image":
      self.embed_s = nn.Sequential(
          # (in - (kernel - 1) - 1) / stride + 1
          # (84 - 8) / 4 + 1 = 20
          nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False),
          nn.ReLU(),
          # (20 - 4) / 2 + 1 = 9
          nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
          nn.ReLU(),
          # (9 - 3) / 1 + 1 = 7
          nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(7 * 7 * 64, config.embed_dim, bias=False),
          nn.Tanh(),
      )
    else:
      self.embed_s = nn.Linear(in_features=config.state_size,
                               out_features=config.embed_dim, bias=False)
    self.embed_a = nn.Linear(in_features=1, out_features=config.embed_dim, bias=False)
    self.embed_R = nn.Linear(in_features=1, out_features=config.embed_dim, bias=False)
    self.embed_t = nn.Linear(in_features=1, out_features=config.embed_dim, bias=False)

    self.action_linear = nn.Linear(in_features=config.embed_dim, out_features=config.action_size)

    self.blocks = nn.Sequential(*[CasualTransformerBlock(config) for _ in range(config.n_block)])

  def get_embeddings(self, input):
    # input(batch, K, value)
    pos_encoding = self.embed_t(torch.sign(input.timestep / self.config.embed_dim))
    # batch, K, embed_diminput
    if self.config.input_type == "image":
      batch, seq = input.state.size()[:2]
      embed_s = self.embed_s(input.state.reshape(-1, *input.state.shape[2:]))
      embed_s = embed_s.reshape(batch, seq, -1) + pos_encoding
    else:
      embed_s = self.embed_s(input.state) + pos_encoding
    embed_a = self.embed_a(input.action) + pos_encoding
    embed_R = self.embed_R(input.rtg) + pos_encoding

    # interleave tokens as (R_1 , s_1 , a_1 , ... , R_K , s_K )
    # batch, 3K, embed_dim
    return torch.cat((embed_R, embed_s, embed_a), dim=1)

  def forward(self, input: Input):
    embeddings = self.get_embeddings(input)

    # batch, 3K, embed_dim
    output = self.blocks(embeddings)
    # batch, 3K, embed_dim

    batch, K_3, embed_dim = embeddings.size()
    # アクションのみ使う
    # batch, 3K, embed_dim -> batch, 3, K, embed_dim
    action = self.action_linear(output.reshape(batch, 3, -1, embed_dim)[:, -1])
    # batch, K, action_size
    return action


class CasualTransformerBlock(nn.Module):
  def __init__(self, config: Config) -> None:
    super(CasualTransformerBlock, self).__init__()

    self.multi_head_layer = MultiHeadLayer(config)
    self.ffn_layer = FFNLayer(config)

  def forward(self, input):
    out1 = self.multi_head_layer(input)
    return self.ffn_layer(out1)


class MultiHeadLayer(nn.Module):
  def __init__(self, config: Config) -> None:
    super(MultiHeadLayer, self).__init__()

    self.config = config

    self.register_buffer("mask", torch.tril(torch.ones(config.n_features + 1, config.n_features + 1))
                         .view(1, 1, config.n_features + 1, config.n_features + 1))
    self.head_dim = config.embed_dim // config.n_head

    self.query = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
    self.key = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
    self.value = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)

    self.linear = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
    self.norm = nn.LayerNorm(config.embed_dim)

  def forward(self, input):
    batch, K_3, embed_dim = input.size()

    # batch, K_3, embed_dim -> batch, n_head, K_3, head_dim
    q = self.query(input).reshape(batch, self.config.n_head, K_3, self.head_dim)
    k = self.key(input).reshape(batch, self.config.n_head, K_3, self.head_dim)
    v = self.value(input).reshape(batch, self.config.n_head, K_3, self.head_dim)
    # batch, n_head, K_3, head_dim

    # batch, n_head, K_3, head_dim × batch, n_head, head_dim, K
    qk_dot = torch.matmul(q, torch.transpose(k, -1, -2))
    # batch, n_head, K_3, K_3
    masked_qk_dot = qk_dot.masked_fill(self.mask[:, :, :K_3, :K_3] == 0, float('-inf'))
    masked_qk_dot = torch.softmax(masked_qk_dot, dim=-1)
    # batch, n_head, K_3, head_dim
    attention_weight = torch.matmul(masked_qk_dot, v).reshape(batch, K_3, embed_dim)

    #  -> batch, K_3, embed_dim
    # batch, K_3, embed_dim
    out_multi_head = self.linear(attention_weight)
    # batch, K_3 * embed_dim
    out = self.norm(input + out_multi_head)

    return out.reshape(batch, K_3, embed_dim)


class FFNLayer(nn.Module):
  def __init__(self, config: Config) -> None:
    super(FFNLayer, self).__init__()

    self.linear1 = nn.Linear(in_features=config.embed_dim, out_features=config.ffn_dim)
    self.linear2 = nn.Linear(in_features=config.ffn_dim, out_features=config.embed_dim)
    self.norm = nn.LayerNorm(config.embed_dim)

  def forward(self, input):
    out_ffn1 = F.relu(self.linear1(input))
    input_ffn2 = torch.where(out_ffn1 < 0, 0, out_ffn1)
    out_ffn2 = self.linear2(input_ffn2)

    out = self.norm(input + out_ffn2)

    return out
