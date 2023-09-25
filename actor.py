
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import Config

from env import AtariEnv, BatchedEnv, EnvOutput
from meta_controller import MetaController
from shared_data import SharedActorData
from utils import get_beta_table


def actor_loop(ids, log_ids, share_actor_data, config: Config):
  summary_writer = SummaryWriter("logs")

  share_actor_data.shared.get_shared_memory()

  total_steps = 0
  episode_rewards = np.zeros(config.num_env_batches)
  episode_counts = np.zeros(config.num_env_batches, dtype=int)
  meta_controller = MetaController(config, config.num_env_batches)
  meta_index = meta_controller.reset()

  env = BatchedEnv(config)
  states = env.reset()

  beta_table = get_beta_table(config)

  share_actor_data.put_actor_data(
    EnvOutput(
      next_state=states,
      reward=np.zeros(config.num_env_batches),
      done=np.zeros(config.num_env_batches, dtype=bool),
    ),
    meta_index)

  while True:
    total_steps += 1

    actions = share_actor_data.get_action()

    env_output = env.step(actions)
    episode_rewards += env_output.reward

    share_actor_data.put_actor_data(env_output, meta_index)

    indexes = np.where(env_output.done)[0]
    if indexes.size > 0:
      meta_index = meta_controller.update(indexes, episode_counts, episode_rewards)
      for index in indexes:
        env_id = ids[index]
        if env_id in log_ids:
          summary_writer.add_scalar(f"reward/{env_id}", episode_rewards[index], total_steps)
          summary_writer.add_scalar(f"beta/{env_id}", beta_table[meta_index[index]], episode_counts[index])

      episode_counts[indexes] += 1
      episode_rewards[indexes] = 0


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
        summary_writer.add_scalar("reward/eval", episode_rewards[0] / config.eval_update_period, total_steps)
        episode_rewards[indexes] = 0

      meta_index = meta_controller.update(indexes, episode_counts, episode_rewards)
      episode_counts[indexes] += 1
