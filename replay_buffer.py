from numpy import random

import numpy as np
from config import Config
from data_type import DataType
from sum_tree import SumTree
import zstandard as zstd


def replay_loop(transition_queue, sample_queue, priority_queue, config: Config):
  replay_buffer = ReplayBuffer(config)

  while True:
    if not transition_queue.empty():
      loss, transition = transition_queue.get()
      replay_buffer.add(loss, transition)

    if replay_buffer._count > config.replay_buffer_min_size and sample_queue.qsize() < 2:
      sample = replay_buffer.get_minibatch(config.batch_size)
      sample_queue.put(sample)

    if not priority_queue.empty():
      idxs, losses = priority_queue.get()
      for idx, loss in zip(idxs, losses):
        replay_buffer.update(idx, loss)


class ReplayBuffer:
  _e = 0.01
  _a = 0.6
  _beta = 0.4
  _beta_increment_per_sampling = 0.001

  def __init__(self, config: Config):
    self._tree = SumTree(config.replay_buffer_size)
    self._capacity = config.replay_buffer_size
    self._count = 0
    self._prev_count = 0
    self._config = config

    data_type = DataType(config)
    self._transition_dtype = data_type.transition_dtype
    self._transitions = np.empty(config.batch_size, dtype=self._transition_dtype)
    self._indexes = np.empty(config.batch_size, dtype=np.int64)
    self._priorities = np.empty(config.batch_size, dtype=np.float32)
    self._dctx = zstd.ZstdDecompressor()
    self._rng = random.default_rng()

  def _get_priority(self, error):
    return (np.abs(error) + self._e) ** self._a

  def add(self, error, transition):
    p = self._get_priority(error)
    self._tree.add(p, transition)
    self._count += 1
    if self._count <= self._config.replay_buffer_min_size:
      add_acount = self._count // self._config.replay_buffer_add_print_size
      if add_acount > self._prev_count:
        self._prev_count = add_acount
        print(f"wait for start replay: {self._count} / {self._config.replay_buffer_min_size}")

  def get_minibatch(self, batch_size):
    segment = self._tree.total() / batch_size

    self._beta = np.min([1., self._beta + self._beta_increment_per_sampling])

    for i in range(batch_size):
      a = segment * i
      b = segment * (i + 1)
      count = 0

      while True:
        s = self._rng.uniform(a, b)
        (idx, p, bytes) = self._tree.get(s)
        if bytes != 0:
          break
        else:
          count += 1
          if count > 10:
            raise ValueError("sample get errors.")

      self._transitions[i] = np.frombuffer(self._dctx.decompress(bytes), dtype=self._transition_dtype)
      self._indexes[i] = idx
      self._priorities[i] = p

    sampling_probabilities = self._priorities / self._tree.total()
    is_weight = np.power(self._tree.n_entries * sampling_probabilities, -self._beta)

    return self._indexes, self._transitions, is_weight / is_weight.max()

  def update(self, idx, error):
    p = self._get_priority(error)
    self._tree.update(idx, p)
