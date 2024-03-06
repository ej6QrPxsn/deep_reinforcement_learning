from atari.atari_env import AtariEnv
from atari.atari_data_writer import AtariDataWriter
from config import Config


class TargetManager:
  def __init__(self, config: Config) -> None:
    if config.target_type == "atari":
      self.writer = AtariDataWriter
      self.env = AtariEnv
