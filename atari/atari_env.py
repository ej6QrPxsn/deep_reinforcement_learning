

import collections
import cv2
from env import Env, EnvOutput
import gymnasium
import numpy as np


def preprocess_frame(frame):
  image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  img1 = image_gray[34: 160 + 34, 0: 160]
  image_resize = cv2.resize(img1, dsize=(84, 84))

  return image_resize


class AtariEnv(Env):
  def __init__(self, env_name, max_step) -> None:
    self.env = gymnasium.make(env_name)
    self.action_space = self.env.action_space.n

    self.frames = None
    self.next_state = None
    self.count = 0
    self.max_step = max_step

  def __reset(self):
    #: ゲーム画面を二値化したりトリミングしたりする前処理
    frame = preprocess_frame(self.env.reset()[0])
    #: DQNでは直近4フレームの集合をstateとする
    self.frames = collections.deque(
        [frame] * 4, maxlen=4)
    return np.stack(self.frames, axis=0)

  def reset(self):
    next_state = self.__reset()
    return EnvOutput(
      next_state=next_state,
      reward=0,
      done=False,
    )

  def step(self, action) -> EnvOutput:
    next_frame, reward, terminated, truncated, info = self.env.step(action)
    self.count += 1

    done = terminated or truncated or self.count >= self.max_step
    self.frames.append(preprocess_frame(next_frame))

    # エピソード終了
    if done:
      next_state = self.__reset()
      self.count = 0
    else:
      next_state = np.stack(self.frames, axis=0)

    return EnvOutput(next_state, reward, done)
