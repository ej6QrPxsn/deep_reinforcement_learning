# SEED RL + NGU + Meta Controller

## 紹介

SEED RLでのNGU（Rescaleなし）をPytorchで実装しています。また、以下を追加で実装しています。

* Meta Controller（Agent57）
* Meta Controllerで選択するβとγの値算出(NGU）

### 参照論文

* [SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference](https://openreview.net/pdf?id=rkgvXlrKwH)
* [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038)(NGU)
* [Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/pdf/2003.13350)

## 実行準備

```
pip install -r requirements.txt
```

## 実行

```
python main.py
```

