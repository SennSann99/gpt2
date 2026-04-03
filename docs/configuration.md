# Configuration (設定概要)

`gpt2/config.py` では、モデルの構造を決める `ModelConfig` と、学習処理を管理する `TrainConfig` を定義している。共にPython標準の `@dataclass` デコレータを用いており、クリーンで明快なパラメータ管理を実現している。

## パフォーマンスの最適化 (`slots=True`)
Python 3.10で導入された機能である `slots=True` が使用されている。通常、Pythonはクラスのインスタンス属性を動的なディクショナリである `__dict__` に保存するが、`slots=True` を明示することで「定義されている変数のみを利用する」とロックすることができる。
これにより、余分な辞書の生成が抑えられ、**メモリの節約とアクセス速度の向上**が実現される。

## ModelConfig

モデル（GPT-2）のサイズおよび構造に関する設定群である。

```python
@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False
```

| パラメータ | 型 | デフォルト値 | 説明 |
|---|---|---|---|
| `vocab_size` | `int` | 50257 | トークナイザ（tiktoken想定）の語彙サイズ |
| `block_size` | `int` | 256 | アテンションが一度に参照可能なコンテキストウィンドウ（最大系列長） |
| `n_layer` | `int` | 12 | Transformerブロックの積み重ね層数 |
| `n_head` | `int` | 12 | マルチヘッドアテンションのヘッドの数 |
| `n_embd` | `int` | 768 | トークンの埋め込みベクトルの次元数 |
| `dropout` | `float`| 0.1 | 過学習防止のためのドロップアウト率 |
| `bias` | `bool` | False | 全結合層や出力層でのバイアスの有無 |

## TrainConfig

データの読み込み先や学習率、エポック数といった訓練過程に関する設定群である。

```python
@dataclass(slots=True)
class TrainConfig:
    data_path: str = "data/Papers.csv"
    text_column: str = "PaperText"
    limit_rows: int = 0
    val_rows: int = 20
    batch_size: int = 2
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    seed: int = 1337
    num_workers: int = 0
    amp: bool = True
    checkpoint_path: str = "checkpoints/last.ckpt"
```

| パラメータ | 型 | デフォルト値 | 説明 |
|---|---|---|---|
| `data_path` | `str` | "data/Papers.csv" | データセット(CSVファイル)のパス |
| `text_column` | `str` | "PaperText" | アサインするCSVデータのテキストを含む列名 |
| `limit_rows` | `int` | 0 | 読み込む行数の制限値(0の場合は全てを読む) |
| `val_rows` | `int` | 20 | 検証用(Validation)に確保するために末尾から切り出す行数 |
| `batch_size` | `int` | 2 | 学習と評価に使用するバッチサイズ |
| `max_steps` | `int` | 1000 | トレーニングの最大ステップ数 |
| `eval_interval` | `int` | 100 | 学習中に評価を実施するステップ間隔 |
| `eval_batches` | `int` | 10 | 評価時に使用されるバッチの数 |
| `learning_rate`| `float`| 3e-4 | 最適化のための学習率（Learning Rate）のベース値 |
| `weight_decay` | `float`| 0.1 | Optimizer(AdamW)で用いる L2正則化の重み減衰割合 |
| `beta1` | `float`| 0.9 | AdamW オプティマイザの beta1 値 |
| `beta2` | `float`| 0.95 | AdamW オプティマイザの beta2 値 |
| `grad_clip` | `float`| 1.0 | 勾配爆発を防ぐためのClipping値 |
| `warmup_steps` | `int` | 100 | 学習率を指定のステップ数まで線形に上昇させるためのウォームアップ設定 |
| `seed` | `int` | 1337 | 再現性確保のためのランダムシード |
| `num_workers` | `int` | 0 | DataLoaderで使用するプロセス（ワーカー）の数 |
| `amp` | `bool`| True | Automatic Mixed Precision(自動混合精度学習)の有効・無効化 |
| `checkpoint_path` | `str` | "checkpoints/last.ckpt" | 実行結果とその時点の重みを保存・ロードするパス |
