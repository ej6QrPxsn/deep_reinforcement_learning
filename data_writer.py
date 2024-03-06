import json
import os
from pathlib import Path
import numpy as np
import webdataset as wds
import datetime as dt


class DataWriter:
  def __init__(self, id, config) -> None:
    self.count = 0
    self.validate_count = 0
    self.id = id
    self._config = config

    self.rng = np.random.default_rng()

    self.shard_writer = wds.ShardWriter(
        pattern=f"file:{config.train_data_dir}/{id}_{config.train_filename}",
        maxsize=config.shard_size,
        verbose=0,
    )

    self.data_dir = self._config.train_data_dir
    self.env_name = self._config.env_name

  def write_data(self, data):
    now = dt.datetime.now()
    time = now.strftime(f"{self.env_name}_%Y%m%d-%H%M%S-%f")
    key_str = time

    self.shard_writer.write({
        "__key__": key_str,
        "bytes": data,
    })
    self.count += 1

  def write_end(self):
    dataset_size_filename = f"{self.data_dir}/{id}_dataset-size.json"
    with open(dataset_size_filename, 'w') as fp:
      json.dump({
          "dataset size": self.count,
      }, fp)

    self.shard_writer.close()

  def info_from_json(self):
    with open(Path(self.data_dir) / f"{self.id}_dataset-size.json", 'r') as f:
      info_dic = json.load(f)
    return int(info_dic['dataset size'])
