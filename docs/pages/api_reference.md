# API Reference

このページでは、プロジェクト内の主要なモジュールに含まれるクラス、関数、および変数の一覧をまとめる。

## gpt2/config.py

モデルと訓練の構成を管理するためのデータクラスが定義されている。

### ModelConfig (dataclass)

| 名前 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | `50257` | 語彙数。 |
| `block_size` | `int` | `512` | 会話履歴を含む最大の系列長。 |
| `n_layer` | `int` | `12` | Transformer Block の層数。 |
| `n_head` | `int` | `12` | マルチヘッドアテンションのヘッド数。 |
| `n_embd` | `int` | `768` | 埋め込みベクトルの次元数。 |
| `dropout` | `float` | `0.1` | ドロップアウト率。 |
| `bias` | `bool` | `False` | 線形層などにバイアスを含めるかどうか。 |

### TrainConfig (dataclass)

| 名前 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `limit_rows` | `int` | `0` | 読み込む行数の制限（0は無制限）。 |
| `val_rows` | `int` | `20` | 検証用データの行数。 |
| `batch_size` | `int` | `2` | バッチサイズ。 |
| `max_steps` | `int` | `-1` | 無制限訓練。`Ctrl+C`で停止。 |
| `eval_interval` | `int` | `100` | 検証の評価間隔（ステップ数）。 |
| `eval_batches` | `int` | `10` | 検証時に使用するバッチ数。 |
| `learning_rate` | `float` | `3e-4` | 学習率。 |
| `weight_decay` | `float` | `0.1` | 重み減衰率。 |
| `beta1` | `float` | `0.9` | Adam の beta1。 |
| `beta2` | `float` | `0.95` | Adam の beta2。 |
| `grad_clip` | `float` | `1.0` | 勾配クリッピング。 |
| `warmup_steps` | `int` | `100` | ウォームアップステップ数。 |
| `seed` | `int` | `1337` | ランダムシード。 |
| `num_workers` | `int` | `0` | DataLoader のワーカー数。 |
| `amp` | `bool` | `True` | 自動混合精度（AMP）の使用フラグ。 |
| `checkpoint_path` | `str` | `"checkpoints"` | チェックポイントの保存先。 |

---

## gpt2/model.py

モデルの構成要素と推論・訓練のロジックが定義されている。

### クラス

| 名前 | 説明 |
| :--- | :--- |
| `RotaryPositionalEmbedding` | Rotary Position Embedding（RoPE）を計算するモジュール。 |
| `CausalSelfAttention` | 因果的セルフアテンション機構。 |
| `MLP` | Position-wise Feed-Forward Network。 |
| `Block` | Transformer Block（LayerNorm + Attention + MLP）。 |
| `GPTModel` | GPT-2 本体のモデル。 |
| `GPTLightning` | `LightningModule` を継承した、学習のライフサイクルを管理するモジュール。 |

### 関数

| 名前 | 説明 |
| :--- | :--- |
| `_rotate_half(x)` | RoPE 計算用の補助関数。 |
| `apply_rotary_emb(x, cos, sin)` | Tensor に RoPE を適用する。 |
| `build_optimizer(model, cfg)` | AdamW オプティマイザを構築する。 |
| `lr_for_step(step, cfg)` | 指定ステップにおける学習率（ウォームアップ対応）を計算する。 |

---

## gpt2/train.py

モデルの訓練を実行するエントリポイント。

### 関数

| 名前 | 説明 |
| :--- | :--- |
| `parse_args()` | コマンドライン引数をパースし、`Config` クラスを生成する。 |
| `tokenize_conversation(messages, max_length)` | マルチターン会話をトークン化し、Assistant部分の学習ラベルを作成する。 |
| `train(model_cfg, train_cfg)` | PyTorch Lightning を用いた訓練プロセスを開始する。 |
| `main()` | `parse_args` と `train` を順に呼び出す。 |

---

## gpt2/generate.py

学習済みモデルを使用してテキストを生成する。

### 関数

| 名前 | 説明 |
| :--- | :--- |
| `parse_args()` | 生成用の引数をパースする。 |
| `generate(model_cfg, train_cfg, prompt, max_new_tokens)` | チェックポイントからモデルをロードし、テキストを生成する。 |
| `main()` | 引数をパースして生成を実行する。 |
