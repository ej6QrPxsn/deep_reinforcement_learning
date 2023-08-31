

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
from typing import NamedTuple

import torch


class ParallelInferenceTask(NamedTuple):
  input_queue: Queue
  output_queue: Queue
  thread: threading.Thread


class InferenceNetworks(NamedTuple):
  infer: torch.nn.Module
  rnd: torch.nn.Module
  embedding: torch.nn.Module


class ParallelTask():
  def __init__(self) -> None:
    self.executor = ThreadPoolExecutor()
    self.future_reward = None

  def inference(self, ids, agent_input, reward_generator, infer_net, RND_net, embedding_net):
    future_infer = self.executor.submit(infer_net, agent_input)
    reward_input = agent_input.state.squeeze(1)
    future_random = self.executor.submit(RND_net.random, reward_input)
    future_predict = self.executor.submit(RND_net.predict, reward_input)
    future_embedding = self.executor.submit(embedding_net, reward_input)

    embedding = future_embedding.result()
    rnd_loss = RND_net.get_loss(future_random.result(), future_predict.result())

    future_reward = self.executor.submit(
      reward_generator.get_intrinsic_reward,
      ids,
      rnd_loss.detach().cpu().numpy(),
      embedding.detach().cpu().numpy()
    )

    return future_infer.result(), future_reward

  def train(self, transitions, RND_net, embedding_net):
    self.executor.submit(RND_net.train, transitions)
    self.executor.submit(embedding_net.train, transitions)

  def add_replay(self, fn, ret, device, config, transition_queue, cctx):
    self.executor.submit(fn, ret, device, config, transition_queue, cctx)
