import random

import numpy as np
from config import Config
from sum_tree import SumTree
import zstandard as zstd


def replay_loop(transition_queue, sample_queue, priority_queue, config: Config):
  replay_buffer = ReplayBuffer(config)

  while True:
    if not transition_queue.empty():
      loss, transition = transition_queue.get()
      replay_buffer.add(loss, transition)

    if replay_buffer.count > config.replay_buffer_min_size and sample_queue.qsize() < 2:
      sample = replay_buffer.get_minibatch(config.batch_size)
      sample_queue.put(sample)

    if not priority_queue.empty():
      idxs, losses = priority_queue.get()
      for idx, loss in zip(idxs, losses):
        replay_buffer.update(idx, loss)


class ReplayBuffer:
  e = 0.01
  a = 0.6
  beta = 0.4
  beta_increment_per_sampling = 0.001

  def __init__(self, config: Config):
    self.tree = SumTree(config.replay_buffer_size)
    self.capacity = config.replay_buffer_size
    self.count = 0
    self.prev_count = 0
    self.config = config
    self.transitions = np.empty(config.batch_size, dtype=self.config.transition_dtype)
    self.indexes = np.empty(config.batch_size, dtype=np.int64)
    self.priorities = np.empty(config.batch_size, dtype=np.float32)
    self.dctx = zstd.ZstdDecompressor()

  def _get_priority(self, error):
    return (np.abs(error) + self.e) ** self.a

  def add(self, error, transition):
    p = self._get_priority(error)
    self.tree.add(p, transition)
    self.count += 1
    if self.count <= self.config.replay_buffer_min_size:
      add_acount = self.count // self.config.replay_buffer_add_print_size
      if add_acount > self.prev_count:
        self.prev_count = add_acount
        print(f"wait for start replay: {self.count} / {self.config.replay_buffer_min_size}")

  def get_minibatch(self, batch_size):
    segment = self.tree.total() / batch_size

    self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

    for i in range(batch_size):
      a = segment * i
      b = segment * (i + 1)
      count = 0

      while True:
        s = random.uniform(a, b)
        (idx, p, bytes) = self.tree.get(s)
        if bytes != 0:
          break
        else:
          count += 1
          if count > 10:
            raise ValueError("sample get errors.")

      self.transitions[i] = np.frombuffer(self.dctx.decompress(bytes), dtype=self.config.transition_dtype)
      self.indexes[i] = idx
      self.priorities[i] = p

    sampling_probabilities = self.priorities / self.tree.total()
    is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)

    return self.indexes, self.transitions, is_weight / is_weight.max()

  def update(self, idx, error):
    p = self._get_priority(error)
    self.tree.update(idx, p)
