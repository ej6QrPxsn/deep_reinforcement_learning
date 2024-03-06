import gc
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
import pickle
import zstandard as zstd
import numpy as np
from config import Config
from data_writer import DataWriter
from local_buffer import Transition


class AtariDataWriter:
  def __init__(self, id, config: Config) -> None:
    self.id = id
    self._config = config

    buffers = np.array_split(np.arange(50), config.n_writers)
    self.suffix_list = buffers[id]
    self.total_steps = 0

    self._rng = np.random.default_rng()

  def write(self):
    ctx = zstd.ZstdCompressor()
    writer = DataWriter(self.id, self._config)
    for suffix in self.suffix_list:
      tmp_buffer = OutOfGraphReplayBuffer(
          replay_capacity=10000,
          observation_shape=(84, 84),
          stack_size=4,
          batch_size=64,
          update_horizon=1,
          gamma=0.99)
      tmp_buffer.load("./data/Breakout/1/replay_logs", suffix=f"{suffix}")

      idx = 0

      indices = [i for i in range(0, tmp_buffer.cursor()) if tmp_buffer.is_valid_transition(i)]
      transitions = tmp_buffer.sample_transition_batch(batch_size=len(indices), indices=indices)
      states, actions, rewards, _, _, _, dones, indices = transitions
      terminal_indices = np.argwhere(dones == 1).flatten().tolist()

      start = 0
      for terminal_idx in terminal_indices:
        if terminal_idx - start > self._config.max_timestep:
          end = start + self._config.max_timestep
        else:
          end = terminal_idx + 1

        for _ in range(start, end):
          idx += 1
          self.total_steps += 1

          data = pickle.dumps(
            Transition(
                state=ctx.compress(states[idx].transpose(2, 0, 1).tobytes()),
                action=actions[idx],
                reward=rewards[idx],
                done=dones[idx]
            )
          )
          writer.write_data(data)

        start = terminal_idx + 1

      del tmp_buffer
      gc.collect()

    writer.write_end()
