# 簡易版 GPT-2 (PyTorch Lightning)

## 更新履歴

| 日付 | 更新内容 |
|---|---|
| 2026-03-15 | 位置エンコーディングを Rotary Position Embedding (RoPE) に置き換えました． |

大学院生レベルのAI研究向けに設計された，最小限かつ研究志向のGPT-2実装です．内部構造が明確で，再現性のある実験が可能であり，PyTorch Lightningを利用したクリーンなスケーリングパスを備えています．

---

## モチベーション

- 可読性の高い最小限のコードを通じて，GPT-2のコアメカニズムを学習する．
- クリーンなデータクラスの構成と直感的なCLIにより，迅速なイテレーションを回す．
- 学習ループを書き換えることなく，マルチGPUや分散学習への拡張性を維持する．

---

## 機能

| コンポーネント | 詳細 |
|---|---|
| モデル | `einsum` を用いた因果的自己注意機構（causal self-attention）を備えたGPT-2スタイルのTransformer |
| 設定 | モデルおよび学習のハイパーパラメータ用の型付きデータクラス |
| 学習 | PyTorch Lightningの `Trainer` + `LightningModule` |
| 最適化 | パラメータグループごとのWeight Decayと線形ウォームアップを備えたAdamW |
| トークン化 | `tiktoken` (GPT-2 エンコーディング) |
| ロギング | `logs/` へのCSVロガー出力．TensorBoard互換 |
| 再現性 | 決定論的なシード設定 |

---

## プロジェクト構成

```text
gpt2/
├── config.py      # モデルと学習設定のデータクラス
├── data.py        # CSV → トークンチャンク → DataLoader
├── model.py       # GPTモデル + LightningModule
├── train.py       # Lightning Trainer エントリポイント (推奨)
└── generate.py    # テキスト生成 エントリポイント
main.py            # 便利なエントリポイント (学習 + 生成)
```

---

## 要件

- Python ≥ 3.12
- [`uv`](https://github.com/astral-sh/uv) (推奨) または `pip`
- PyTorch ≥ 2.5, Lightning ≥ 2.6

依存関係のインストール:

```bash
uv sync
```

---

## クイックスタート

```bash
uv run python -m gpt2.train \
  --max-steps 200 \
  --eval-interval 20 \
  --batch-size 2
```

### スモークテスト (迅速な動作確認)

```bash
uv run python -m gpt2.train \
  --limit-rows 40 \
  --block-size 64 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 128 \
  --batch-size 2 \
  --max-steps 2 \
  --eval-interval 1 \
  --eval-batches 1
```

---

## データフォーマット

- `PaperText` 列を持つCSVファイルを想定しています（例：学術論文の要旨や全文）．
- デフォルトのデータセットパス: `data/Papers.csv`

---

## 出力物

| 生成物 | パス |
|---|---|
| 最新のチェックポイント | `checkpoints/last.ckpt` |
| 学習ログ | `logs/` |

---

## Dockerでの実行 (run.sh)

### 方法1: run.sh を使用する (推奨)

Dockerを利用して、ビルドから実行（GPU・tmux対応）までを自動化するスクリプトを用意しています。

```bash
# 基本的な実行 (CPU)
./run.sh

# GPUを使用して実行
./run.sh gpu

# tmuxセッション内で実行
./run.sh tmux

# GPUを使用し、かつtmuxセッション内で実行
./run.sh gpu tmux
```

- **自動削除**: コンテナ終了時に `--rm` オプションによりコンテナは自動的に削除されます。
- **永続化**: `logs/`, `checkpoints/`, `data/` ディレクトリはホスト側にマウントされ、データが保持されます。

### 方法2: 手動でコマンドを実行する

```bash
# ビルド
docker build -t gpt2 .

# GPUを使用して実行 (終了時に自動削除)
docker run --rm -it --gpus all gpt2
```

---

## ドキュメントの閲覧

プロジェクトの詳細や研究ノートを確認するためのローカルドキュメントサーバーが用意されています。

```bash
./open_docs.sh
```

- **ポート**: デフォルトで 8080 番を使用します。
- **自動起動**: スクリプトを実行すると、自動的にブラウザでドキュメントが開きます。自動で開かない場合は、ブラウザで [http://localhost:8080](http://localhost:8080) に直接アクセスしてください。
- **キャッシュ無効化**: 常に最新の内容を確認できるよう、キャッシュを無効化した状態でサーバーが起動します。

---

## 並列処理

学習はPyTorch Lightning上に構築されているため，単一のCPU/GPUからマルチGPUのDDP（Distributed Data Parallel）へのスケーリングにおいて，学習ループを変更する必要はありません．Lightningがデバイスの配置，DDPのセットアップ，および自動混合精度（Mixed Precision）を自動的に処理します．