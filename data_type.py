import numpy as np

from config import Config


class DataType():
  def __init__(self, config: Config):
    # シーケンス蓄積用遷移データ
    self.work_transition_dtype = np.dtype([
        ("state", "u1", config.state_shape),
        ("action", "u1"),
        ("reward", "f4"),
    ])
