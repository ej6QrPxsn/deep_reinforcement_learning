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
          nn.Conv2d(4, 32, kernel_size=8, stride=4),
          nn.ReLU(),
          # (20 - 4) / 2 + 1 = 9
          nn.Conv2d(32, 64, kernel_size=4, stride=2),
          nn.ReLU(),
          # (9 - 3) / 1 + 1 = 7
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(7 * 7 * 64, config.embed_dim),
          nn.Tanh(),
      )
    else:
      self.embed_s = nn.Sequential(nn.Linear(in_features=config.state_size,
                                             out_features=config.embed_dim), nn.Tanh())
    self.embed_a = nn.Sequential(nn.Embedding(config.action_size, config.embed_dim), nn.Tanh())
    self.embed_R = nn.Sequential(nn.Linear(in_features=1, out_features=config.embed_dim), nn.Tanh())

    self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.embed_dim))
    self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.embed_dim))

    self.action_linear = nn.Linear(in_features=config.embed_dim, out_features=config.action_size, bias=False)

    self.blocks = nn.Sequential(*[CasualTransformerBlock(config, device) for _ in range(config.n_block)])
    self.drop = nn.Dropout(config.embed_drop)

    self.action_norm = nn.LayerNorm(config.embed_dim)

  def get_embeddings(self, input):
    state = input.state / 255.

    if self.config.input_type == "image":
      batch, seq = state.size()[:2]
      embed_s = self.embed_s(state.reshape(-1, *state.shape[2:]))
      embed_s = embed_s.reshape(batch, seq, -1)
    else:
      embed_s = self.embed_s(state)
    embed_a = self.embed_a(input.action)
    embed_R = self.embed_R(input.rtg)

    embed_token = torch.empty(embed_s.shape[0], embed_s.shape[1] * 3, embed_s.shape[2]).to(self.device)
    embed_token[:, 0::3, :] = embed_R
    embed_token[:, 1::3, :] = embed_s
    embed_token[:, 2::3, :] = embed_a

    # global_pos_embをバッチ方向に拡張
    # batch, max_timestep, embed_dim
    batch_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, embed_s.shape[0], dim=0)

    # timestepをエンベッド方向に拡張
    # batch, 1, embed_dim
    timestep_emb = torch.repeat_interleave(input.timestep, self.config.embed_dim, dim=-1)

    # global_pos_embから、timestep位置にある値を取得する
    # batch, 1, embed_dim
    global_timestep_pos_emb = torch.gather(batch_global_pos_emb, 1, timestep_emb)

    # pos_embをトークン長さに制限
    # 1, token_len, embed_dim
    token_len_pos_emb = self.pos_emb[:, :embed_token.shape[1], :]

    pos_encoding = global_timestep_pos_emb + token_len_pos_emb

    return embed_token + pos_encoding

  def forward(self, input: Input):
    batch, seq = input.state.shape[:2]
    embeddings = self.get_embeddings(input)

    # batch, 3K, embed_dim
    output = self.blocks(self.drop(embeddings))
    # batch, 3K, embed_dim

    output = self.action_norm(output)

    # 状態のみ使う
    # アクションだと学習できない
    # 状態でも学習できないらしい
    # batch, 3K, embed_dim -> batch, K, embed_dim
    logits = self.action_linear(output[:, 1::3, :])
    # batch, K, action_size

    return logits


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

    self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                         .view(1, 1, config.block_size + 1, config.block_size + 1))
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
