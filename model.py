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
  timestep: torch.Tensor


class DecisionTransformer(nn.Module):
  def __init__(self, config: Config, device) -> None:
    super(DecisionTransformer, self).__init__()

    self.config = config
    self.device = device

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
      self.embed_s = nn.Sequential(nn.Linear(in_features=config.state_size,
                                             out_features=config.embed_dim, bias=False), nn.Tanh())
    self.embed_a = nn.Sequential(nn.Linear(in_features=1, out_features=config.embed_dim, bias=False), nn.Tanh())
    self.embed_R = nn.Sequential(nn.Linear(in_features=1, out_features=config.embed_dim, bias=False), nn.Tanh())

    self.pos_emb = nn.Parameter(torch.zeros(1, config.context_length * 3 + 1, config.embed_dim))
    self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.embed_dim))

    self.action_linear = nn.Linear(in_features=config.embed_dim, out_features=config.action_size)

    self.blocks = nn.Sequential(*[CasualTransformerBlock(config, device) for _ in range(config.n_block)])
    self.drop = nn.Dropout(config.embed_drop)

    self.action_norm = nn.LayerNorm(config.embed_dim)

  def get_embeddings(self, input):

    if self.config.input_type == "image":
      batch, seq = input.state.size()[:2]
      embed_s = self.embed_s(input.state.reshape(-1, *input.state.shape[2:]))
      embed_s = embed_s.reshape(batch, seq, -1)
    else:
      embed_s = self.embed_s(input.state)
    embed_a = self.embed_a(input.action)
    embed_R = self.embed_R(input.rtg)

    embed_token = torch.empty(embed_s.shape[0], embed_s.shape[1] * 3, embed_s.shape[2]).to(self.device)
    embed_token[:, 0::3, :] = embed_R
    embed_token[:, 1::3, :] = embed_s
    embed_token[:, 2::3, :] = embed_a

    batch_size = input.state.size()[0]

    all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)  # batch_size, traj_length, n_embd

    pos_encoding = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(input.timestep, self.config.embed_dim, dim=-1)) + self.pos_emb[:, :embed_token.shape[1], :]

    return embed_token + pos_encoding

  def forward(self, input: Input):
    embeddings = self.get_embeddings(input)

    # batch, 3K, embed_dim
    output = self.blocks(self.drop(embeddings))
    # batch, 3K, embed_dim

    batch, K_3, embed_dim = embeddings.size()

    output = self.action_norm(output)

    # アクションのみ使う
    # batch, 3K, embed_dim -> batch, K, embed_dim
    action_out = output[:, 2::3, :]

    action = self.action_linear(action_out)
    # batch, K, action_size

    return action


class CasualTransformerBlock(nn.Module):
  def __init__(self, config: Config, device) -> None:
    super(CasualTransformerBlock, self).__init__()

    self.multi_head_layer = MultiHeadLayer(config, device)
    self.ffn_layer = FFNLayer(config)
    self.norm1 = nn.LayerNorm(config.embed_dim)
    self.norm2 = nn.LayerNorm(config.embed_dim)

  def forward(self, x):
    x = x + self.multi_head_layer(x)
    x = self.norm1(x)
    x = x + self.ffn_layer(x)
    x = self.norm2(x)

    return x


class MultiHeadLayer(nn.Module):
  def __init__(self, config: Config, device) -> None:
    super(MultiHeadLayer, self).__init__()

    self.config = config

    self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length))
                         .view(1, 1, config.context_length, config.context_length))
    self.head_dim = config.embed_dim // config.n_head

    self.query = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
    self.key = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
    self.value = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)

    self.linear = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)

    self.mask_value = torch.tensor(float("-inf")).to(torch.float16).to(device)

  def forward(self, input):
    batch, K_3, embed_dim = input.size()

    # batch, K_3, embed_dim -> batch, n_head, K_3, head_dim
    q = self.query(input).reshape(batch, self.config.n_head, K_3, self.head_dim)
    k = self.key(input).reshape(batch, self.config.n_head, K_3, self.head_dim)
    v = self.value(input).reshape(batch, self.config.n_head, K_3, self.head_dim)
    # batch, n_head, K_3, head_dim

    # batch, n_head, K_3, head_dim × batch, n_head, head_dim, K
    x = torch.matmul(q, torch.transpose(k, -1, -2))
    # batch, n_head, K_3, K_3
    x = x.masked_fill(self.mask[:, :, :K_3, :K_3] == 0, self.mask_value)
    x = torch.softmax(x, dim=-1)
    # batch, n_head, K_3, head_dim
    x = torch.matmul(x, v).reshape(batch, K_3, embed_dim)

    #  -> batch, K_3, embed_dim
    # batch, K_3, embed_dim
    x = self.linear(x)
    # batch, K_3 * embed_dim

    return x.reshape(batch, K_3, embed_dim)


class FFNLayer(nn.Module):
  def __init__(self, config: Config) -> None:
    super(FFNLayer, self).__init__()

    self.linear1 = nn.Linear(in_features=config.embed_dim, out_features=config.ffn_dim)
    self.linear2 = nn.Linear(in_features=config.ffn_dim, out_features=config.embed_dim)

  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = torch.where(x < 0, 0, x)
    out = self.linear2(x)

    return out
