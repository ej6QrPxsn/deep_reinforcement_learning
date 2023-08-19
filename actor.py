
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import Config

from env import AtariEnv, BatchedEnv, EnvOutput
from meta_controller import MetaController
from shared_data import SharedEnvData


def actor_loop(ids, log_ids, share_env_data, config: Config):
  summary_writer = SummaryWriter("logs")

  share_env_data.shared.get_shared_memory()

  total_steps = 0
  episode_rewards = np.zeros(config.num_env_batches)
  meta_controller = MetaController(config)
  betas, gammas = meta_controller.reset()

  env = BatchedEnv(config)
  states = env.reset()

  share_env_data.put_env_data(
    EnvOutput(
      next_state=states,
      reward=np.zeros(config.num_env_batches),
      done=np.zeros(config.num_env_batches, dtype=bool),
    ),
    betas, gammas)

  while True:
    total_steps += 1

    actions = share_env_data.get_action()

    env_output = env.step(actions)
    episode_rewards += env_output.reward

    share_env_data.put_env_data(env_output, betas, gammas)

    betas, gammas = meta_controller.update(episode_rewards)
    indexes = np.where(env_output.done)[0]
    if indexes.size > 0:
      for index in indexes:
        env_id = ids[index]
        if env_id in log_ids:
          summary_writer.add_scalar(f"reward/{env_id}", episode_rewards[index], total_steps)

      episode_rewards[indexes] = 0


def tester_loop(share_env_data: SharedEnvData, config: Config):
  summary_writer = SummaryWriter("logs")
  share_env_data.shared.get_shared_memory()

  total_steps = 0
  episode_reward = 0

  env = AtariEnv(config.env_name)
  states = env.reset()

  beta = np.zeros((1, 1))
  beta[:] = config.eval_epsilon

  share_env_data.put_env_data(
    EnvOutput(
      next_state=states,
      reward=0,
      done=False,
    ),
    beta, None)

  while True:
    total_steps += 1

    action = share_env_data.get_action()

    env_output = env.step(action[0])
    episode_reward += env_output.reward

    share_env_data.put_env_data(env_output, beta, None)

    indexes = np.where(env_output.done)[0]
    if indexes.size > 0:
      summary_writer.add_scalar("reward/eval", episode_reward, total_steps)

      episode_reward = 0
