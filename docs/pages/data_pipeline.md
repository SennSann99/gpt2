# Data Pipeline (データパイプライン)

`gpt2/data.py` は、プレーンなテキストデータ（具体的にはCSVファイル）を、モデルが解釈可能なPyTorchテンソルのミニバッチに変換する一連の処理を担っている。

## 全体の流れ

1. CSVファイルからのテキスト読み込み
2. `tiktoken` を用いたテキスト次元から一次元のトークンIDの羅列への変換（結合）
3. Token Chunking (ブロックサイズごとの切り出し)
4. PyTorch `Dataset` および `DataLoader` への変換
5. `LightningDataModule` による管理と供給

---

### トークンへの変換 (`_flatten_texts_to_tokens`)

CSVファイルで指定された列（デフォルトは `PaperText`）のテキストを順次読み込む。
その際、OpenAIによって開発された高速なパブリックトークナイザである `tiktoken` (モデル設定 "gpt2") を使用してテキストをエンコード処理する。

```python
def _flatten_texts_to_tokens(texts, tokenizer) -> torch.Tensor:
    eot = tokenizer.eot_token # 文末や次の文章への切れ目を表す end-of-text
    ids: list[int] = []
    
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        # tiktokenを用いて文字列を整数の配列に変換し、1つの大きなリストに結合
        ids.extend(tokenizer.encode_ordinary(text))
        ids.append(eot)
        
    return torch.tensor(ids, dtype=torch.long)
```

テキストは独立したデータではあるものの、文脈の連続性を保ちつつパディングや余白の無駄を排除するために、一度すべてのトークンを **大きな一次元のリスト（1Dテンソル）に結合** する。各文章の末尾には「終了トークン」（`<|endoftext|>` = `eot_token`）が挿入され、区切りを表現するように設計されている。

### チャンク生成 (`TokenChunkDataset`)

学習するために、一次元に繋がった巨大なトークン列を固定長（`block_size`）に切り出す必要がある。PyTorch標準の `Dataset` クラスを継承し、以下の仕様で入力テンソル ($x$) と ラベルテンソル ($y$) を作成する:

```python
class TokenChunkDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int, stride: int | None = None):
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride or block_size
        self.n_chunks = max(0, (len(token_ids) - 1 - block_size) // self.stride + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        # x : 指定ブロックサイズ分のトークン抽出
        x = self.token_ids[start : start + self.block_size]
        # y : 次の単語を当てるため、xから1単語分だけ後ろにずらしたトークン抽出
        y = self.token_ids[start + 1 : start + self.block_size + 1]
        return x, y
```

- $x$ (入力): インデックス位置 `start` から `start + block_size` までのトークン。
- $y$ (ターゲット): 次のトークンを予測するため、入力全体を1要素分先にずらした `start + 1` から `start + block_size + 1` までのトークン。

### 学習と検証の分割 (`build_dataloaders`)

CSVに含まれるすべての行（テキスト群）に対して、トークン変換を行うまえのPandas DataFrameの状態で「学習用」と「検証用 (Validation)」の分割を行う。引数 `limit_rows` で全体の読み込み件数を絞ったり、`val_rows` によって検証用に最後に回す行数を決定することができる。

分割した後に、トークン化およびチャンク生成を行い、最終的に `DataLoader` へとラップさせる。環境に応じて、マルチプロセッシング機能である `num_workers` や、GPUへの転送速度を高める `pin_memory` オプションを適切に有効化する。

### LightningDataModule (`GPTDataModule`)

これらの構築プロセスは `pl.LightningDataModule` を継承する `GPTDataModule` を起点に呼び出される。モデル側である `LightningModule` と同様に、データ処理の前段階に関わる一連の動作をカプセル化している。

- `setup(stage)` 関数において上記のデータセットおよび DataLoaderが作られ、インスタンス化される。
- `train_dataloader()` および `val_dataloader()` を実装することで、PyTorch LightningのTrainer環境へシームレスにデータ供給を行う。
