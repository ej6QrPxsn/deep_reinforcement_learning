import numpy as np
from config import Config
from meta_controller import MetaController


def test_get_max_arg_index():
  config = Config()

  config.num_env_batches = 3
  config.num_arms = 32
  config.bandit_window_size = 90
  config.bandit_UCB_beta = 1
  config.bandit_epsilon = 0.5

  meta_controller = MetaController(config)
  ids = np.array([0, 1, 2])

  steps = np.array([2, 20, 120])

  # ID0のアーム15の報酬合計2
  # ID0のアーム25の報酬合計6

  # ID1のアーム15の報酬合計14
  # ID1のアーム25の報酬合計8

  # ID2のアーム15の報酬合計4
  # ID2のアーム25の報酬合計10
  meta_controller._reward_table[ids, 0, 15] = np.array([1, 7, 2])
  meta_controller._reward_table[ids, 3, 15] = np.array([1, 7, 2])
  meta_controller._reward_table[ids, 0, 25] = np.array([3, 4, 5])
  meta_controller._reward_table[ids, 3, 25] = np.array([3, 4, 5])

  values = meta_controller._get_max_arg_index(ids, steps)
  expected = np.array([25, 15, 25])

  np.testing.assert_array_equal(values, expected)


def test_get_arm_index():
  config = Config()

  config.num_env_batches = 3
  config.num_arms = 32
  config.epsilon_beta = 1 - 1e-6
  # 乱数分岐に行かない設定
  config.bandit_epsilon = 0

  meta_controller = MetaController(config)
  ids = np.array([0, 1, 2])

  steps = np.array([50, 70, 120])

  # ID0のアーム15の報酬合計2
  # ID0のアーム25の報酬合計6

  # ID1のアーム15の報酬合計14
  # ID1のアーム25の報酬合計8

  # ID2のアーム15の報酬合計4
  # ID2のアーム25の報酬合計10
  meta_controller._reward_table[ids, 0, 15] = np.array([1, 7, 2])
  meta_controller._reward_table[ids, 3, 15] = np.array([1, 7, 2])
  meta_controller._reward_table[ids, 0, 25] = np.array([3, 4, 5])
  meta_controller._reward_table[ids, 3, 25] = np.array([3, 4, 5])

  values = meta_controller._get_arm_index(ids, steps)
  expected = np.array([25, 15, 25])

  np.testing.assert_array_equal(values, expected)
