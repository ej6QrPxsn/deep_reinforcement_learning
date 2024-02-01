import json
from pathlib import Path
import numpy as np
import webdataset as wds
import datetime as dt


def last_write_data(id, config, train_count, validate_count):
  dataset_size_filename = f"{config.train_data_dir}/{id}_dataset-size.json"
  with open(dataset_size_filename, 'w') as fp:
    json.dump({
        "dataset size": train_count,
    }, fp)

  dataset_size_filename = f"{config.validate_data_dir}/dataset-size.json"
  with open(dataset_size_filename, 'w') as fp:
    json.dump({
        "dataset size": validate_count,
    }, fp)


def info_from_json(id, shard_dir):
  with open(Path(shard_dir) / f"{id}_dataset-size.json", 'r') as f:
    info_dic = json.load(f)
  return int(info_dic['dataset size'])


class DataWriter:
  def __init__(self, id, config) -> None:
    self.train_count = 0
    self.validate_count = 0
    self.config = config
    self.id = id

    self.rng = np.random.default_rng()

    self.train_writer = wds.ShardWriter(
        pattern=f"file:{config.train_data_dir}/{id}_{config.train_filename}",
        maxsize=config.shard_size,
        verbose=0,
    )
    self.validate_writer = wds.ShardWriter(
        pattern=f"file:{config.validate_data_dir}/{id}_{config.validate_filename}",
        maxsize=config.shard_size,
        verbose=0,
    )

  def write_train_data(self, data):
    now = dt.datetime.now()
    time = now.strftime("%Y%m%d-%H%M%S-%f")
    key_str = time

    if self.rng.random() < self.config.train_date_ratio:
      self.train_writer.write({
          "__key__": key_str,
          "bytes": data,
      })
      self.train_count += 1
    else:
      self.validate_writer.write({
          "__key__": key_str,
          "bytes": data,
      })
      self.validate_count += 1

  def write_end(self):
    last_write_data(self.id, self.config, self.train_count, self.validate_count)
    self.train_writer.close()
    self.validate_writer.close()
