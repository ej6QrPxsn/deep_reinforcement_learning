

import time
import numpy as np
from tqdm import tqdm

import zstandard as zstd


class ReplayBuffer:
  def __init__(self, config, data_type, sample_queue) -> None:
    self.data_list = []
    self.empty_list = list(range(config.replay_size))
    self.data = np.empty(config.replay_size, dtype=object)

    self.sample_queue = sample_queue
    self.config = config
    self.data_type = data_type
    self.rng = np.random.default_rng()

    self.fill = False
    self.sample_data = np.empty(config.batch_size, dtype=data_type.transition_dtype)

  def _add(self, data):
    # 最大サイズの場合、最も古いデータを上書きする
    if len(self.data_list) == self.config.replay_size:
      index = self.data_list[-1]
      self.data[index] = data
      self.fill = True
    else:
      index = self.empty_list.pop(0)
      self.data[index] = data
      self.data_list.append(index)

  def replay_loop(self, load_queue):
    dctx = zstd.ZstdDecompressor()
    bar = tqdm(total=self.config.min_replay_size, position=2)
    bar.set_description('replay')

    count = 0

    while True:
      data = load_queue.get()
      if data is None:
        count += 1
        if count == self.config.n_loads:
          break

      self._add(data)
      if bar:
        if len(self.data_list) <= self.config.min_replay_size:
          bar.update(1)
        else:
          bar.close()
          bar = None

      if self.sample_queue.empty():
        if self.fill:
          self._put_sample(dctx)
        # バッファのデータ数が最小サイズ以上なら
        elif not self.fill and len(self.data_list) >= self.config.min_replay_size:
          # サンプル設定
          self._put_sample(dctx)

    # ファイルからの読み込みは終わっている
    # データがなくなるまでサンプル設定する
    while len(self.data_list) > 0:
      if self.sample_queue.empty():
        # サンプル設定
        self._put_sample(dctx)
      else:
        time.sleep(1)

    self.sample_queue.put(None)

  def _put_sample(self, dctx):
    if self.sample_queue.empty():
      # バッファのデータ数が最小サイズ以上なら
      if len(self.data_list) >= self.config.min_replay_size:
        # サンプル取得
        self._sample(dctx)
      # バッファのデータ数が最小サイズ未満なら
      else:
        self._sample(dctx)

  def _sample(self, dctx):
    # データ数がバッチ数より大きいなら
    if len(self.data_list) > self.config.batch_size:
      # バッチ数のサンプルを取り出す
      batch_size = self.config.batch_size
    # データ数がバッチ数未満なら
    else:
      # データ数のサンプルを取り出す
      batch_size = len(self.data_list)

    # データ数が最小リプレイサイズより大きいなら
    if len(self.data_list) > self.config.min_replay_size:
      min_replay_size = self.config.min_replay_size
    else:
      min_replay_size = len(self.data_list)

    indexes = self.rng.integers(0, min_replay_size, batch_size)
    for i, index in enumerate(indexes):
      # バッファから解凍して取得
      self.sample_data[i] = np.frombuffer(dctx.decompress(self.data[index], max_output_size=self.data_type.transition_dtype.itemsize),
                                          dtype=self.data_type.transition_dtype)
      # 取得インデクスはデータリストから削除
      self.data_list.pop(0)
      # 空きリストに追加
      self.empty_list.append(index)

    self.sample_queue.put(self.sample_data)
