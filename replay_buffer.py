

from collections import deque
import itertools
import time
import numpy as np
from tqdm import tqdm

import zstandard as zstd
from config import Config

from local_buffer import EpisodeTransition


class ReplayBuffer:

  def __init__(self, config: Config, data_type) -> None:
    self._config = config
    self._seq_len = config.context_length

    self._transition_dtype = data_type.transition_dtype

    self._rng = np.random.default_rng()

    self._states = deque(maxlen=config.replay_size)
    self._actions = deque(maxlen=config.replay_size)
    self._rtgs = deque(maxlen=config.replay_size)
    self._timesteps = deque(maxlen=config.replay_size)
    self._dones = deque(maxlen=config.replay_size)

    self._ready = False

    self._state_dtype = np.dtype([
        ("state", "u1", config.state_shape),
    ])

  def replay_loop(self, load_queue, sample_queue):
    dctx = zstd.ZstdDecompressor()
    bar = tqdm(total=self._config.min_replay_size, position=2)
    bar.set_description('replay')

    load_complete_count = 0
    train_count = 0

    while True:
      data = load_queue.get()
      if data is None:
        load_complete_count += 1
        if load_complete_count == self._config.n_loads:
          break

      self._add_data(data)
      if bar:
        if len(self._states) <= self._config.min_replay_size:
          bar.n = len(self._states)
          bar.refresh()
        else:
          self._ready = True
          bar.close()
          bar = None

      if sample_queue.empty():
        if self._ready:
          train_count += 1
          d = self._sample(dctx, self._config.batch_size)
          sample_queue.put(d)

    # ファイルからの読み込みは終わっている
    # 訓練ステップ数までサンプル取得する
    while train_count < self._config.train_steps:
      if sample_queue.empty():
        train_count += 1
        sample_queue.put(self._local_buffer.sample(self._config.batch_size))
      else:
        time.sleep(1)

    self.sample_queue.put(None)

  def _add_data(self, transition: EpisodeTransition):
    add_len = transition.timestep
    timesteps = np.arange(add_len)
    dones = np.zeros(add_len, dtype=bool)
    dones[-1] = True

    self._states.extend(transition.states)
    self._actions.extend(transition.actions)
    self._rtgs.extend(transition.rtgs)
    self._timesteps.extend(timesteps)
    self._dones.extend(dones)

  def _sample(self, dctx: zstd.ZstdDecompressor, batch_size):
    sample_data = np.zeros(batch_size, dtype=self._transition_dtype)
    indexes = self._rng.integers(0, len(self._states) - self._seq_len, batch_size)
    for i, index in enumerate(indexes):
      start = index
      end = start + self._seq_len

      # シーケンス終わり確定
      dones = list(itertools.islice(self._dones, start, end))
      done_indexes = np.where(np.array(dones))[0]
      if done_indexes:
        end = done_indexes[0]
        start = end - self._seq_len
        if start < 0:
          start = 0

      # シーケンス開始確定
      dones = list(itertools.islice(self._dones, start, end))
      done_indexes = np.where(np.array(dones))[0]
      if done_indexes:
        start = done_indexes[-1]

      bytes_list = list(itertools.islice(self._states, start, end))

      states = []
      for bytes in bytes_list:
        state = np.frombuffer(dctx.decompress(bytes), dtype=np.uint8)
        states.append(state.reshape(self._config.state_shape))

      states = np.stack(states)

      sample_data[i]["state"][:end - start] = states
      sample_data[i]["action"][:end - start] = list(itertools.islice(self._actions, start, end))
      sample_data[i]["rtg"][:end - start] = list(itertools.islice(self._rtgs, start, end))
      sample_data[i]["timestep"] = self._timesteps[start]

    return sample_data
