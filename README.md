# Decision Transformer

## 紹介

Decision TransformerをPytorchで実装しています。

### 参照論文

* [Decision Transformer: Outperforming the Atari Human Benchmark](https://arxiv.org/abs/2106.01345)

### 公式実装
* [Decision Transformer](https://github.com/kzl/decision-transformer/tree/master)

## 実行準備

```
cd .devcontainer
docker-compose -f docker-compose.yml up -d
docker run --name drl -it drl /bin/bash
```
または、devcontainerを使う場合は、devcontainerをインストールして起動

## 実行

```
python3 train.py
```

## 結果

