# SEED RL + Agent57

## 紹介

SEED RL + Agent57をPytorchで実装しています。

### 参照論文

* [SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference](https://openreview.net/pdf?id=rkgvXlrKwH)
* [Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/pdf/2003.13350)

## 実行準備

```
pip install -r requirements.txt
```

## 実行

```
python main.py
```

## 結果

### 実行環境

Python 3.10 / Ryzen 5950x / RTX A4000

#### 設定

* Atari Game："Breakout"
* アクター数：16
* 1アクターあたり環境数：16
* 訓練環境数：256
* 推論プロセス数：4

### ステップ数

 ![image](images/breakout_step.png)

### 実行時間

 ![image](images/breakout_time.png)
