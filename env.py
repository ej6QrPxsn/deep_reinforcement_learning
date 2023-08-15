

import collections
from typing import NamedTuple
import cv2
import gymnasium
import numpy as np

from config import Config


class EnvOutput(NamedTuple):
  next_state: np.ndarray
  reward: np.ndarray
  done: np.ndarray
  transition_done: np.ndarray


def preprocess_frame(frame):
  image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  img1 = image_gray[34: 160 + 34, 0: 160]
  image_resize = cv2.resize(img1, dsize=(84, 84))

  return image_resize


class Env:
  def __init__(self, env_name) -> None:
    self.action_space = None

  def reset(self):
    pass

  def step(self, action) -> EnvOutput:
    pass


class AtariEnv(Env):
  def __init__(self, env_name) -> None:
    self.env = gymnasium.make(env_name)
    self.action_space = self.env.action_space.n

    self.lives = 5
    self.frames = None
    self.next_state = None

  def reset(self):
    self.lives = 5

    #: ゲーム画面を二値化したりトリミングしたりする前処理
    frame = preprocess_frame(self.env.reset()[0])
    #: DQNでは直近4フレームの集合をstateとする
    self.frames = collections.deque(
        [frame] * 4, maxlen=4)
    return np.stack(self.frames, axis=0)

  def step(self, action) -> EnvOutput:
    next_frame, reward, terminated, truncated, info = self.env.step(action)
    done = terminated or truncated
    self.frames.append(preprocess_frame(next_frame))

    #: ライフが減ったら経験上はゲーム終了扱いとする
    if info["lives"] != self.lives:
      self.lives = info["lives"]
      transition_done = True
    else:
      transition_done = done

    # エピソード終了
    if done:
      next_state = self.reset()
    else:
      next_state = np.stack(self.frames, axis=0)

    return EnvOutput(next_state, reward, done, transition_done)


class BatchedEnv(Env):
  def __init__(self, config: Config) -> None:
    self.envs = np.zeros(config.num_env_batches, dtype=object)
    for i in range(config.num_env_batches):
      self.envs[i] = AtariEnv(config.env_name)

    self.next_states = np.empty((config.num_env_batches, *config.state_shape))
    self.rewards = np.zeros(config.num_env_batches)
    self.transition_dones = np.zeros(config.num_env_batches, dtype=bool)
    self.dones = np.zeros(config.num_env_batches, dtype=bool)

  def step(self, actions) -> EnvOutput:
    for i, (env, action) in enumerate(zip(self.envs, actions)):
      output = env.step(action)
      self.next_states[i] = output.next_state
      self.rewards[i] = output.reward
      self.transition_dones[i] = output.transition_done
      self.dones[i] = output.done

    return EnvOutput(
        next_state=self.next_states,
        reward=self.rewards,
        transition_done=self.transition_dones,
        done=self.dones,
    )

  def reset(self):
    for i, env in enumerate(self.envs):
      self.next_states[i] = env.reset()
    return self.next_states
