import math
from typing import List, NamedTuple
import torch
import torch.nn.functional as F

from config import Config


class InputWeights(NamedTuple):
  query: torch.nn.Linear
  key: torch.nn.Linear
  value: torch.nn.Linear


class Input(NamedTuple):
  reward: torch.Tensor
  state: torch.Tensor
  action: torch.Tensor
  timestep: int


class DecisionTransformer(torch.nn.Module):
  def __init__(self, config: Config) -> None:
    super(DecisionTransformer, self).__init__()

    self.config = config
    self.embed_s = torch.nn.Linear(in_features=config.state_size,
                                   out_features=config.model_dimension, bias=False)
    self.embed_a = torch.nn.Linear(in_features=1, out_features=config.model_dimension, bias=False)
    self.embed_R = torch.nn.Linear(in_features=1, out_features=config.model_dimension, bias=False)
    self.embed_t = torch.nn.Linear(in_features=1, out_features=config.model_dimension, bias=False)

    self.action_linear = torch.nn.Linear(in_features=config.model_dimension, out_features=config.action_size)

    self.transformer = CasualTransformer(config)

  def get_embeddings(self, input):
    # input(batch, seq, value)
    pos_encoding = self.embed_t(torch.sign(input.timestep / self.config.model_dimension))
    # batch, seq, model_dimension
    embed_s = self.embed_s(input.state) + pos_encoding
    embed_a = self.embed_a(input.action) + pos_encoding
    embed_R = self.embed_R(input.reward) + pos_encoding

    # interleave tokens as (R_1 , s_1 , a_1 , ... , R_K , s_K )
    # batch, seq * 3, model_dimension
    return torch.cat((embed_R, embed_s, embed_a), dim=1)

  def forward(self, input: Input):
    embeddings = self.get_embeddings(input)

    # batch, seq * 3, model_dimension
    output = self.transformer(embeddings)

    split_output = torch.chunk(output, 3, dim=1)
    # 3(R, s, a), batch, seq, model_dimension

    # アクションのみ使う
    # batch, seq, model_dimension
    action = self.action_linear(split_output[-1])
    # batch, seq, action_size
    return action


class CasualTransformer(torch.nn.Module):
  def __init__(self, config: Config) -> None:
    super(CasualTransformer, self).__init__()

    self.config = config

    self.blocks = []
    for _ in range(config.num_stack_block):
      self.blocks.append(CasualTransformerBlock(config))

  def forward(self, embeddings):
    for block in self.blocks:
      embeddings = block.forward(embeddings)
    return embeddings


class CasualTransformerBlock(torch.nn.Module):
  def __init__(self, config: Config) -> None:
    super(CasualTransformerBlock, self).__init__()

    self.multi_head_layer = MultiHeadLayer(config)
    self.ffn_layer = FFNLayer(config)

  def forward(self, input):
    out1 = self.multi_head_layer(input)
    return self.ffn_layer(out1)


class MultiHeadLayer(torch.nn.Module):
  def __init__(self, config: Config) -> None:
    super(MultiHeadLayer, self).__init__()

    self.config = config
    self.multi_head_dimension = config.model_dimension // config.num_head
    self.multi_head_dimension_scale = torch.tensor(math.sqrt(config.model_dimension))

    self.weights: List[InputWeights] = []
    for i in range(config.num_head):
      weights = InputWeights(
          query=torch.nn.Linear(
              in_features=self.multi_head_dimension, out_features=self.multi_head_dimension),
          key=torch.nn.Linear(
              in_features=self.multi_head_dimension, out_features=self.multi_head_dimension),
          value=torch.nn.Linear(
              in_features=self.multi_head_dimension, out_features=self.multi_head_dimension),
      )
      self.weights.append(weights)

    self.linear = torch.nn.Linear(
        in_features=config.model_dimension, out_features=config.model_dimension)
    self.norm = torch.nn.LayerNorm(config.model_dimension)

  def get_attention_weight(self, weights, input):
    q = weights.query(input)
    k = weights.key(input)
    v = weights.value(input)

    # batch, seq * 3(r, s, a), multi_head_dimension × batch, multi_head_dimension, seq * 3(r, s, a)
    attention_weight = torch.matmul(q, torch.transpose(k, 2, 1))
    # batch, seq * 3(r, s, a), seq * 3

    # 最後のアクションだけ重みがゼロになるようにする
    attention_weight[:, -1, -self.config.seq_len:] = -float('inf')

    attention_weight = torch.softmax(attention_weight, dim=2) / self.multi_head_dimension_scale
    # batch, seq * 3, seq * 3 × batch, seq * 3(r, s, a), multi_head_dimension
    attention_weight = torch.matmul(attention_weight, v)
    # batch, seq * 3(r, s, a), multi_head_dimension
    return attention_weight

  def forward(self, input):
    o_list = []
    # 入力をマルチヘッド数に分割してアテンション取得
    multi_inputs = torch.chunk(input, self.config.num_head, dim=2)
    # batch, seq * 3, multi_head_dimension
    for multi_input, weights in zip(multi_inputs, self.weights):
      attention_weight = self.get_attention_weight(weights, multi_input)
      o_list.append(attention_weight)

    # 出力を統合して、元の次元数に戻す
    concat_out = torch.cat(o_list, dim=2)
    # batch, seq * 3(r, s, a), model_dimension

    out_multi_head = self.linear(concat_out)
    out = self.norm(input + out_multi_head)

    return out


class FFNLayer(torch.nn.Module):
  def __init__(self, config: Config) -> None:
    super(FFNLayer, self).__init__()

    self.linear1 = torch.nn.Linear(in_features=config.model_dimension, out_features=config.ffn_dimension)
    self.linear2 = torch.nn.Linear(in_features=config.ffn_dimension, out_features=config.model_dimension)
    self.norm = torch.nn.LayerNorm(config.model_dimension)

  def forward(self, input):
    out_ffn1 = F.relu(self.linear1(input))
    input_ffn2 = torch.where(out_ffn1 < 0, 0, out_ffn1)
    out_ffn2 = self.linear2(input_ffn2)

    out = self.norm(input + out_ffn2)

    return out
