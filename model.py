from typing import List, NamedTuple
import numpy as np
import torch
import torch.nn.functional as F

from config import Config

class InputWeights(NamedTuple):
  query: torch.nn.Linear
  key: torch.nn.Linear
  value: torch.nn.Linear

class MultiHeadSubLayer(NamedTuple):
  weights: List[InputWeights]
  linear: torch.nn.Linear
  norm: torch.nn.LayerNorm

class FFNSubLayer(NamedTuple):
  linear1: torch.nn.Linear
  linear2: torch.nn.Linear
  norm: torch.nn.LayerNorm

class Input(NamedTuple):
  reward: torch.Tensor
  state: torch.Tensor
  action: torch.Tensor
  start_timestep: int
  end_timestep: int


class DecisionTransformer(torch.nn.Module):
  def __init__(self, config: Config) -> None:
    self.config = config
    self.embed_s = torch.nn.Linear(in_features=config.state_size, out_features=config.model_dimension, bias=False)
    self.embed_a = torch.nn.Linear(in_features=config.seq_len, out_features=config.model_dimension, bias=False)
    self.embed_R = torch.nn.Linear(in_features=config.seq_len, out_features=config.model_dimension, bias=False)
    self.embed_t = torch.nn.Linear(in_features=config.seq_len, out_features=config.model_dimension, bias=False)

    self.action_linear = torch.nn.Linear(in_features=config.model_dimension, out_features=config.action_size)

    self.transformer = CasualTransformer(config)

  def forward(self, input: Input):
    # linearは入力の最後の次元がfeatureと同じなら問題ない
    # input(batch, seq, value)
    pos_encoding = self.get_positional_encode(input)
    # batch, seq, model_dimension
    embed_s = self.embed_s(input.state) + pos_encoding
    embed_a = self.embed_a(input.action) + pos_encoding
    embed_R = self.embed_R(input.reward) + pos_encoding

    # interleave tokens as (R_1 , s_1 , a_1 , ... , R_K , s_K )
    # 3, batch, seq, model_dimension
    embeddings = torch.stack((embed_s, embed_a, embed_R), dim=2).permute(1, 2, 0, 3)
    # batch, seq, 3, model_dimension
    output = self.transformer(embeddings)

    # batch, seq, 3, model_dimension
    # アクションのみ使う
    pre_action = output[:, :, 1, :]
    # batch, seq, model_dimension
    action = self.action_linear(pre_action)
    # batch, seq, action_size

  def get_positional_encode(self, input: Input):
    # batch, seq
    pos_input = torch.empty(input.reward.shape[:2])
    timestep = range(input.start_timestep, input.end_timestep)
    pos_input[:] = torch.sign(timestep / self.config.model_dimension)
    return self.embed_t(pos_input)


class CasualTransformer:
  def __init__(self, config: Config) -> None:

    self.num_stack_block = config.num_stack_block
    # batch, seq, 3, model_dimension
    self.mask = torch.ones(config.batch_size, config.seq_len, 3, config.model_dimension)
    for i in range(config.seq_len):
      # 位置iはアクションのみ無効化
      # embeddings = torch.stack((embed_s, embed_a, embed_R), dim=2)
      self.mask[:, i, 1, :] = -float('inf')
      # 位置i以降の値はすべて無効化
      self.mask[:, i + 1:, :, :] = -float('inf')

    self.blocks = []
    self.multi_dimension = config.model_dimension / config.num_head
    self.multi_dimension_scale = torch.sqrt(config.multi_dimension)
    for _ in range(config.num_stack_block):
      multi_head_weights: List[InputWeights] = []
      for i in range(config.num_head):
        weights = InputWeights(
          query = torch.nn.Linear(in_features=self.multi_dimension, out_features=self.multi_dimension),
          key = torch.nn.Linear(in_features=self.multi_dimension, out_features=self.multi_dimension),
          value = torch.nn.Linear(in_features=self.multi_dimension, out_features=self.multi_dimension),
        )
        multi_head_weights.append(weights)

      multi_head_linear = torch.nn.Linear(in_features=config.model_dimension, out_features=config.model_dimension)

      multi_head_norm = torch.nn.LayerNorm(config.model_dimension)

      ffn1 = torch.nn.Linear(in_features=config.model_dimension, out_features=config.ffn_dimension)
      ffn2 = torch.nn.Linear(in_features=config.ffn_dimension, out_features=config.model_dimension)

      ffn_norm = torch.nn.LayerNorm(config.model_dimension)

      multi_head_sub_layer = MultiHeadSubLayer(
        weights=multi_head_weights,
        linear=multi_head_linear,
        norm=multi_head_norm,
      )

      ffn_sub_layer = FFNSubLayer(
        linear1=ffn1,
        linear2=ffn2,
        norm=ffn_norm,
      )

      self.blocks.append(multi_head_sub_layer, ffn_sub_layer)

  def forward(self):
    for i in range(self.config.num_stack_block):
      embeddings = self.forward_block(i, embeddings)
    return embeddings

  def forward_block(self, i, input):
    multi_head_sub_layer, ffn_sub_layer = self.blocks[i]
    o_list = []

    # 入力をマルチヘッド数に分割してアテンション取得
    # batch, seq, 3, *model_dimension*
    multi_inputs = torch.split(input, self.multi_dimension, dim=3)
    for i, (multi_input, w) in enumerate(zip(multi_inputs, multi_head_sub_layer.weights)):
      q = w.query(multi_input)
      # 長さKの場合、i時点で知り得るのはiの報酬と状態まで
      # iのアクションや、i+以後の状態、報酬、アクション、つまり未来を知り得てはいけない
      # 未来の情報は全て－∞でマスクすることで、アテンションの重みはゼロとなる
      k = w.key(multi_input) * self.mask
      v = w.value(multi_input)

      # scaled dot product attention
      out_dot_product_attention = torch.softmax(torch.dot(q, k) / self.multi_dimension_scale) * v

      o_list.append(out_dot_product_attention)

    # 出力を統合して、元の次元数に戻す
    concat_o = torch.cat(o_list, dim=0)

    out_multi_head = multi_head_sub_layer.linear(concat_o)

    input_ffn = multi_head_sub_layer.norm(input + out_multi_head)

    out_ffn1 = F.relu(ffn_sub_layer.linear1(input_ffn))
    out_ffn2 = ffn_sub_layer.linear2(torch.max(0, out_ffn1))

    out_ffn_norm = ffn_sub_layer.norm(input_ffn + out_ffn2)

    return out_ffn_norm

