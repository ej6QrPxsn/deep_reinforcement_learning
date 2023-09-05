
from dataclasses import dataclass
import traceback

import numpy as np
import torch

from config import Config
import faiss


@dataclass
class WelfordData():
  count: np.ndarray
  mean: np.ndarray
  M2: np.ndarray

  def reset(self, env_ids):
    self.count[env_ids] = 0
    self.mean[env_ids] = 0
    self.M2[env_ids] = 0


def welford_update(prev_data, new_value, env_ids):
  """https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  For a new value newValue, compute the new count, new mean, the new M2.
  mean accumulates the mean of the entire dataset
  M2 aggregates the squared distance from the mean
  count aggregates the number of samples seen so far
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

  Args:
      prev_data (WelfordData): 前の計算結果
      new_value (float): 新しい値
      env_ids (List[int]): 計算対象の環境

  Returns:
      WelfordData: 新しい計算結果
  """
  prev_data.count[env_ids] += 1
  delta = new_value - prev_data.mean[env_ids]
  with np.errstate(divide='raise'):
    try:
      prev_data.mean[env_ids] += delta / prev_data.count[env_ids]
    except Exception:
      print(prev_data)
      traceback.print_stack()

  delta2 = new_value - prev_data.mean[env_ids]
  prev_data.M2[env_ids] += delta * delta2
  return prev_data


# Retrieve the mean, variance and sample variance from an aggregate
def welford_finalize(prev_data, env_ids):
  mean = prev_data.mean[env_ids]
  variance = prev_data.M2[env_ids] / (prev_data.count[env_ids]).astype(float)
  count = np.where(prev_data.count[env_ids] == 1, [2] * len(env_ids), prev_data.count[env_ids])
  sample_variance = prev_data.M2[env_ids] / (count - 1).astype(float)
  return mean, variance, sample_variance


class RewardGenerator():
  def __init__(self, batch_size, config: Config, embedding_net, device) -> None:
    self.device = device
    self.episodic_novelty = EpisodicNovelty(batch_size, config, embedding_net)

  def get_intrinsic_reward(self, ids, values):
    values = torch.from_numpy(values.squeeze(1).copy()).to(torch.float32).to(self.device)
    return self.episodic_novelty.get_episodic_intrinsic_reward(ids, values)

  def reset(self, ids):
    self.episodic_novelty.reset(ids)


class EpisodicMemory():
  def __init__(self, batch_size, config: Config) -> None:
    self.num_kernel = config.num_kernel

    self.neighbours = np.zeros((batch_size, config.num_kernel), dtype=np.float32)

    self.faiss_indexes = np.empty(batch_size, dtype=object)
    for i in range(batch_size):
      self.faiss_indexes[i] = faiss.IndexFlatL2(config.controllable_state_size)

  def add(self, env_ids, values):
    for env_id in env_ids:
      self.faiss_indexes[env_id].add(values[env_id][np.newaxis, :])

  def reset(self, env_ids):
    for env_id in env_ids:
      self.faiss_indexes[env_id].reset()

  def get_nearest_neighbours(self, env_ids, values):
    for env_id in env_ids:
      distances, indexes = self.faiss_indexes[env_id].search(values[env_id][np.newaxis, :], k=self.num_kernel)
      # 見つからなかった場合の距離を0で取得する
      no_indexes = np.where(indexes == -1)[0]
      distances[no_indexes] = 0
      self.neighbours[env_id] = distances

    return self.neighbours


class EpisodicNovelty():
  def __init__(self, batch_size, config: Config, embedding_net) -> None:
    self.episodic_memory = EpisodicMemory(batch_size, config)
    self.prev_cal_data = WelfordData(
      count=np.zeros((batch_size, config.num_kernel), dtype=np.int32),
      mean=np.zeros((batch_size, config.num_kernel), dtype=np.float32),
      M2=np.zeros((batch_size, config.num_kernel), dtype=np.float32),
    )
    self.batch_size = batch_size
    self.cluster_distance = config.kernel_cluster_distance
    self.epsilon = config.kernel_epsilon
    self.pseudo_counts_constant = config.kernel_pseudo_counts_constant
    self.maximum_similarity = config.kernel_maximum_similarity

    self.embedding_net = embedding_net

  def reset(self, env_ids):
    self.episodic_memory.reset(env_ids)
    self.prev_cal_data.count[env_ids] = 0

  def get_episodic_intrinsic_reward(self, env_ids, value):
    controllable_state = self.embedding_net(value)
    controllable_state = controllable_state.cpu().detach().numpy()
    self.episodic_memory.add(env_ids, controllable_state)

    # env_ids, k
    neighbours = self.episodic_memory.get_nearest_neighbours(env_ids, controllable_state)
    self.prev_cal_data = welford_update(self.prev_cal_data, neighbours, env_ids)
    dn = self.prev_cal_data.mean[env_ids]

    # batch, k
    dn = np.where(dn < self.cluster_distance, 0., dn - self.cluster_distance)
    # batch, k
    Kv = self.epsilon / (dn + self.epsilon)

    # batch
    s = np.sqrt(np.sum(Kv, axis=1)) + self.pseudo_counts_constant

    # batch
    r = np.where(s > self.maximum_similarity, 0., 1 / s)

    return r
