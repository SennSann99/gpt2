<div align="center">

# Compact GPT-2 ⚡

**PyTorch Lightningで学ぶ、研究・実験向けのコンパクトなGPT-2 + Mixture of Experts実装**

[English](README.md) · [ドキュメント](docs/index.html) · [問題解決レポート](docs/problem-report.html)

</div>

---

## このプロジェクトについて

GPT-2の内部構造を読みやすいコードで理解し、学習・評価・文章生成までを一通り試せる研究志向の実装です。PyTorch Lightningを利用し、実験設定、ログ、チェックポイントを整理しながらCPU/GPU環境で実行できます。

### 主な機能

| 項目 | 内容 |
|---|---|
| モデル | GPT-2スタイルのCausal Transformer |
| 位置表現 | Rotary Position Embedding（RoPE） |
| FFN | SwiGLU + Mixture of Experts（MoE） |
| MoEルーティング | Top-k routing、capacity制御、補助損失、Router Z-Loss |
| 学習 | PyTorch Lightning `Trainer` / `LightningModule` |
| データ | Hugging Face Dataset `arman-bd/guppylm-60k-generic` |
| ロギング | CSV + Weights & Biases（任意） |
| チェックポイント | `best.ckpt` / `last.ckpt` / `interrupted.ckpt` |
| UI | Streamlitによる対話型チャット画面 |

---

## 目次

- [最短で実行する](#最短で実行する)
- [必要な環境](#必要な環境)
- [実行方法](#実行方法)
- [学習済みモデルで生成する](#学習済みモデルで生成する)
- [チャットUIを起動する](#チャットuiを起動する)
- [WandBを設定する](#wandbを設定する)
- [チェックポイントとログ](#チェックポイントとログ)
- [プロジェクト構成](#プロジェクト構成)
- [主な設定](#主な設定)
- [トラブルシューティング](#トラブルシューティング)

---

## 最短で実行する

NVIDIA GPUを搭載したLinuxサーバーでは、次のコマンドだけでDockerイメージのビルド、学習、チェックポイントの読み込み、文章生成まで実行できます。

```bash
git clone https://github.com/SennSann99/gpt2.git
cd gpt2
git switch sen/moe
chmod +x run.sh frontend/run_app.sh open_docs.sh
./run.sh gpu
```

WandBも利用する場合は、先に[WandBを設定する](#wandbを設定する)を確認してから実行します。

```bash
./run.sh gpu wandb
```

> [!NOTE]
> 初回実行時はDockerイメージ、GPT-2トークナイザー、学習データセットをダウンロードするため、時間がかかります。

---

## 必要な環境

### Dockerで実行する場合（推奨）

- Git
- Docker
- NVIDIA GPUを使う場合:
  - NVIDIA Driver
  - NVIDIA Container Toolkit

GPUがDockerから認識されることを確認します。

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### ローカルで実行する場合

- Python 3.12以上
- [`uv`](https://docs.astral.sh/uv/)
- PyTorch 2.5以上
- PyTorch Lightning 2.6以上

依存関係をインストールします。

```bash
uv sync --frozen
```

> [!IMPORTANT]
> 学習データとトークナイザーをHugging Faceから取得するため、初回実行時はインターネット接続が必要です。

---

## 実行方法

### 1. `run.sh`で実行する（推奨）

`run.sh`は、Dockerイメージのビルドとコンテナ起動を自動化します。オプションは任意の順序で組み合わせられます。

| オプション | 内容 |
|---|---|
| `gpu` | NVIDIA GPUを有効化します |
| `wandb` | Weights & Biasesへのログ送信を有効化します |
| `tmux` | `gpt2`という名前のtmuxセッション内で実行します |

```bash
# CPUで学習 + 生成
./run.sh

# GPUで学習 + 生成
./run.sh gpu

# GPU + WandB
./run.sh gpu wandb

# GPU + WandB + tmux
./run.sh gpu wandb tmux
```

学習設定もコマンドラインから変更できます。

```bash
./run.sh gpu \
  --max-steps 200 \
  --eval-interval 20 \
  --batch-size 2
```

コンテナ終了後も、次のディレクトリはホスト側に保存されます。

```text
checkpoints/    学習済みモデル
logs/           CSVログ
data/           ローカルデータ用領域
```

### 2. 短い動作確認を行う

モデルと学習パイプラインを少ないステップ数で確認します。

```bash
./run.sh gpu \
  --block-size 64 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 128 \
  --batch-size 2 \
  --max-steps 2 \
  --eval-interval 1 \
  --eval-batches 1
```

### 3. Dockerを使わずに学習する

```bash
uv run python -m gpt2.train \
  --max-steps 200 \
  --eval-interval 20 \
  --batch-size 2
```

### 4. Dockerコマンドを直接使う

```bash
docker build -t gpt2 .

docker run --rm --init -it --gpus all \
  -v "$PWD/logs:/workspace/logs" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  -v "$PWD/data:/workspace/data" \
  gpt2
```

---

## 学習済みモデルで生成する

チェックポイントのディレクトリを指定すると、最新の有効なチェックポイントが自動選択されます。

```bash
uv run python -m gpt2.generate \
  --checkpoint-path checkpoints \
  --prompt "人工知能の未来について"
```

特定のチェックポイントを使う場合は、ファイルを直接指定します。

```bash
uv run python -m gpt2.generate \
  --checkpoint-path checkpoints/version_0/best.ckpt \
  --prompt "人工知能の未来について" \
  --max-new-tokens 128
```

自動選択では、バージョンの新しい順に次の優先順位で検索します。

1. `best.ckpt`
2. `last.ckpt`
3. `interrupted.ckpt`

最新のバージョンディレクトリが空でも、以前の有効なチェックポイントへ自動的にフォールバックします。

---

## チャットUIを起動する

学習後、StreamlitのチャットUIをDockerで起動できます。

```bash
frontend/run_app.sh checkpoints
```

ブラウザで次のURLを開きます。

```text
http://localhost:8501
```

特定のチェックポイントを使用する場合:

```bash
frontend/run_app.sh checkpoints/version_0/best.ckpt
```

> [!NOTE]
> 現在の`frontend/run_app.sh`は`--gpus all`を使用するため、NVIDIA GPUとNVIDIA Container Toolkitが必要です。

---

## WandBを設定する

1. [Weights & Biases](https://wandb.ai/)でアカウントを作成します。
2. WandBの設定画面からAPI Keyを取得します。
3. プロジェクトルートに`.env`を作成します。

```dotenv
WANDB_ENTITY=your_entity_name
WANDB_API_KEY=your_api_key
```

4. `wandb`オプションを付けて実行します。

```bash
./run.sh gpu wandb
```

> [!WARNING]
> `.env`はGitへコミットしないでください。このリポジトリでは`.gitignore`の対象です。

---

## チェックポイントとログ

実行ごとに新しい`version_N`ディレクトリが作成されます。

```text
checkpoints/
├── version_0/
│   ├── best.ckpt
│   └── last.ckpt
└── version_1/
    ├── best.ckpt
    └── last.ckpt

logs/
└── gpt2/
    ├── version_0/
    └── version_1/
```

| ファイル | 内容 |
|---|---|
| `best.ckpt` | `val_loss`が最小になったモデル |
| `last.ckpt` | 学習完了時の最新モデル |
| `interrupted.ckpt` | `Ctrl+C`による中断時に保存されたモデル |

---

## プロジェクト構成

```text
.
├── gpt2/
│   ├── chat.py          # 学習・推論用チャット形式
│   ├── checkpoint.py    # チェックポイントの検索
│   ├── config.py        # モデル・学習設定
│   ├── generate.py      # 文章生成
│   ├── manager.py       # MoE補助損失の管理
│   ├── model.py         # GPT-2、RoPE、SwiGLU、MoE
│   └── train.py         # データ処理と学習
├── frontend/
│   ├── app.py           # Streamlitアプリ
│   ├── Dockerfile
│   └── run_app.sh       # UI起動スクリプト
├── docs/                # ローカルドキュメント
├── checkpoints/         # 学習済みモデル（Git管理外）
├── logs/                # 学習ログ（Git管理外）
├── Dockerfile
├── main.py              # 学習後に文章生成を実行
├── open_docs.sh
├── pyproject.toml
└── run.sh
```

---

## 主な設定

一般的な学習設定はCLIから変更できます。

```bash
uv run python -m gpt2.train --help
```

| 引数 | デフォルト | 内容 |
|---|---:|---|
| `--block-size` | `256` | コンテキスト長 |
| `--n-layer` | `12` | Transformerレイヤー数 |
| `--n-head` | `12` | Attention Head数 |
| `--n-embd` | `768` | 埋め込み次元 |
| `--batch-size` | `8` | バッチサイズ |
| `--max-steps` | `100` | 最大学習ステップ数 |
| `--eval-interval` | `100` | 検証を行う間隔 |
| `--learning-rate` | `3e-4` | 学習率 |
| `--checkpoint-path` | `checkpoints` | 保存先ディレクトリ |
| `--no-amp` | 無効 | Mixed Precisionを無効化 |

MoEの詳細設定は`gpt2/config.py`の`ModelConfig`で管理します。

| 設定 | デフォルト | 内容 |
|---|---:|---|
| `n_exp` | `8` | Expert数。`1`の場合は通常のMLPを使用 |
| `top_k` | `2` | トークンごとに選択するExpert数 |
| `stride` | `2` | MoEレイヤーへ置き換える間隔 |
| `train_capacity` | `1.25` | 学習時のExpert capacity係数 |
| `eval_capacity` | `2.0` | 評価時のExpert capacity係数 |

---

## ローカルドキュメント

```bash
./open_docs.sh
```

起動後、ブラウザで[http://localhost:8080](http://localhost:8080)を開きます。終了するにはターミナルで`Ctrl+C`を押してください。

---

## トラブルシューティング

### チェックポイントが見つからない

保存済みファイルを確認します。

```bash
find checkpoints -maxdepth 2 -type f -name '*.ckpt' -print
```

ファイルが存在する場合は、明示的に指定して生成できます。

```bash
uv run python -m gpt2.generate \
  --checkpoint-path checkpoints/version_0/best.ckpt
```

### DockerからGPUが見つからない

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

このコマンドが失敗する場合は、NVIDIA DriverとNVIDIA Container Toolkitを確認してください。

### WandBが有効にならない

`.env`の設定を確認します。

```bash
grep -E '^(WANDB_ENTITY|WANDB_API_KEY)=' .env
```

API Keyは画面共有、ログ、Issueなどへ貼り付けないでください。

### Git pushがnon-fast-forwardで拒否される

リモートの変更をrebaseで取り込んでからpushします。

```bash
git fetch origin
git rebase origin/sen/moe
git push --set-upstream origin sen/moe
```

競合が発生した場合は、内容を解決してから次を実行します。

```bash
git add -u
git rebase --continue
```

---

## 更新履歴

| 日付 | 更新内容 |
|---|---|
| 2026-03-15 | 位置表現をRotary Position Embedding（RoPE）へ変更 |
| 2026-04-16 | Weights & BiasesによるWeb監視を追加 |
| 2026-07-11 | MoE、チャット形式、チェックポイント自動検索を統合 |

---

<div align="center">

**Read the code. Run the experiment. Understand the model.**

</div>
