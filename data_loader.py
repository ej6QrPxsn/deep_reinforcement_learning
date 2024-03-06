from pathlib import Path
import pickle
import webdataset as wds
from torch.utils.data import DataLoader

from local_buffer import LocalBuffer


class SingleDataLoader:
  def __init__(self, id, config, data_type) -> None:
    self.id = id

    shards_list = [
        str(path) for path in Path(config.train_data_dir).glob(f"{id}_*.tar")
    ]

    self._local_buffer = LocalBuffer(config, data_type)

    self.dataset = wds.WebDataset(shards_list)
    self.dataset = self.dataset.to_tuple("bytes")
    dataloader = DataLoader(self.dataset)
    self.data_it = iter(dataloader)
    self.end = False
    # total = info_from_json(id, config.train_data_dir)

  def load(self, data_queue):

    try:
      byte_data = next(self.data_it)[0][0]

      # ファイルから解凍して読み込み
      data = pickle.loads(byte_data)

      episode_data = self._local_buffer.add_data(data)
      if episode_data:
        data_queue.put(episode_data)
      return True

    except StopIteration:
      self.dataset.close()
      self.end = True
      data_queue.put(None)
      return None
