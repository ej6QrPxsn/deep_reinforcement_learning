
import pickle
from typing import NamedTuple
from data_writer import DataWriter
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import Config

from env import AtariEnv, BatchedEnv, EnvOutput
from meta_controller import MetaController
from shared_data import SharedActorData
from utils import get_beta_table
import zstandard as zstd


class Transition(NamedTuple):
    state: np.ndarray
    action: np.uint8
    reward: np.float32
    done: bool


def actor_loop(ids, log_ids, share_actor_data, config: Config):
    summary_writer = SummaryWriter("logs")

    share_actor_data.shared.get_shared_memory()
    ctx = zstd.ZstdCompressor()

    total_steps = 0
    episode_rewards = np.zeros(config.num_env_batches)
    episode_counts = np.zeros(config.num_env_batches, dtype=int)
    meta_controller = MetaController(config, config.num_env_batches)
    meta_index = meta_controller.reset()

    env = BatchedEnv(config)
    states = env.reset()

    writers = []
    for id in ids:
        writers.append(DataWriter(id, config))

    beta_table = get_beta_table(config)

    share_actor_data.put_actor_data(
        EnvOutput(
            next_state=states,
            reward=np.zeros(config.num_env_batches),
            done=np.zeros(config.num_env_batches, dtype=bool),
        ),
        meta_index)

    for total_steps in range(config.train_steps):
        actions = share_actor_data.get_action()

        env_output = env.step(actions)

        for i in range(config.num_env_batches):
            data = pickle.dumps(
                Transition(
                    ctx.compress(states[i].tobytes()),
                    actions[i],
                    env_output.reward[i],
                    env_output.done[i]
                )
            )
            writers[i].write_train_data(data)

        episode_rewards += env_output.reward
        states[:] = env_output.next_state

        share_actor_data.put_actor_data(env_output, meta_index)

        indexes = np.where(env_output.done)[0]
        if indexes.size > 0:
            meta_index = meta_controller.update(
                indexes, episode_counts, episode_rewards)
            for index in indexes:
                env_id = ids[index]
                if env_id in log_ids:
                    summary_writer.add_scalar(
                        f"reward/{env_id}", episode_rewards[index], total_steps)
                    summary_writer.add_scalar(
                        f"beta/{env_id}", beta_table[meta_index[index]], episode_counts[index])

            episode_counts[indexes] += 1
            episode_rewards[indexes] = 0

    for writer in writers:
        writer.write_end()


def tester_loop(share_actor_data: SharedActorData, config: Config):
    summary_writer = SummaryWriter("logs")
    share_actor_data.shared.get_shared_memory()

    total_steps = 0
    episode_rewards = np.zeros(1)
    episode_counts = np.zeros(1, dtype=int)
    meta_controller = MetaController(config, 1)
    meta_index = meta_controller.reset()

    env = AtariEnv(config.env_name)
    states = env.reset()

    meta_index = 0

    share_actor_data.put_actor_data(
        EnvOutput(
            next_state=states,
            reward=0,
            done=False,
        ),
        meta_index)

    while True:
        total_steps += 1

        action = share_actor_data.get_action()

        env_output = env.step(action[0])
        episode_rewards += env_output.reward

        share_actor_data.put_actor_data(env_output, meta_index)

        indexes = np.where(env_output.done)[0]
        if indexes.size > 0:
            if episode_counts[indexes] % config.eval_update_period == 0:
                summary_writer.add_scalar(
                    "reward/eval", episode_rewards[0] / config.eval_update_period, total_steps)
                episode_rewards[indexes] = 0

            meta_index = meta_controller.update(
                indexes, episode_counts, episode_rewards)
            episode_counts[indexes] += 1
