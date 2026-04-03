# Training & Generation (学習とテキスト生成)

このプロジェクトでは、GPT-2モデルの学習に `gpt2/train.py` を、推論を通じた自立的なテキスト生成に `gpt2/generate.py` を使用する。いずれもコマンドラインインターフェース(CLI)からの直接実行に対応している。

統合の入り口として、プロジェクトのルートレベルにある `main.py` から呼び出すアプローチも可能である。

---

## 訓練の仕組み (`train.py`)

学習ループは PyTorch Lightning の `Trainer` APIを利用して簡素化・堅牢化されている。これによって、従来のPyTorchで行われていた煩雑なループ記述（順伝播、逆伝播、勾配の更新やゼロ化、デバイス配置など）に対する懸念が排除され、研究・実験自体に集中できる仕組みが提供される。

### 1. 引数と設定のパース
Pythonの標準機能である `argparse` を用いて、CLI経由で多数のハイパーパラメータを受け付ける。
パースされた引数は、内部で `ModelConfig` および `TrainConfig` というデータクラスに整理され、モデルとデータローダーに配分される仕組みとなっている。

### 2. インスタンスの初期化と構成
以下の必要なコンポーネントが実体化される。
- 再現性の確保 (`pl.seed_everything`)
- トークナイザー (`tiktoken`)
- データモジュール (`GPTDataModule`)
- モデルモジュール (`GPTLightning`)

### 3. コールバック (Callbacks) とロギング
学習を監視・安定化させるための各種コールバックを設定する。
- **ModelCheckpoint**: 検証損失 (`val_loss`) をモニタリングし、最新のエポック・ステップのチェックポイント（`last.ckpt`）を自動保存する。
- **LearningRateMonitor**: スケジューラの指示に従って変化する学習率をトラッキングする。
- **CSVLogger**: 実験結果や損失の推移などを、 TensorBoard とも完全互換性のある `logs/` ディレクトリ配下に直接CSV形式でシリアライズする。

### 4. Trainer の構築と開始
`pl.Trainer` はハードウェア環境を自動検出 (`accelerator="auto"`, `devices="auto"`) し、Single GPU・Multi-GPU DDPを適切に構成する。
さらに、AMPのフラグに応じて自動混合精度 (`precision="16-mixed"` または `"32-true"`) のモードを選択する。このすべての設定と準備が完了し次第、最後に `trainer.fit()` が呼びさされ学習が開始される。

```python
    # Lightningの力で学習プロセス全体を管理
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=train_cfg.max_steps,
        val_check_interval=train_cfg.eval_interval,
        limit_val_batches=train_cfg.eval_batches,
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step")],
        gradient_clip_val=train_cfg.grad_clip,
        precision=precision,
        log_every_n_steps=1,
    )

    # 用意したモデル構造とデータモジュールを渡して訓練スタート
    trainer.fit(module, datamodule=datamodule)
```

---

## テキストの生成 (`generate.py`)

学習済みのチェックポイント（`.ckpt` ファイル）から状態を復元し、与えられたプロンプトから後続のトークンを反復計算して生成する。推論のみを行うため、`torch.no_grad()` デコレータを付与して勾配計算を安全に無効化し、メモリの使用量と計算コストを節約している。

### 推論の流れ
1. **モデルのロード**: `GPTLightning.load_from_checkpoint` を利用して指定されたチェックポイントから学習済みの状態（重みベクトル）を復元し、評価モード（`.eval()`）に切り替える。
2. **プロンプトのトークン化**: インストールされた `tiktoken` エンコーダを使用して、文字列（例："One day,"）をトークンIDの配列に変換しPyTorchのテンソルとして入力する。
3. **世代ループ (`model.generate()`)**:
   - 入力シーケンスの現在の末尾ブロックサイズ部分だけをスライスし、コンテキストとして利用する。
   - モデルを順伝播させる。生成されたロジットベクトル群（`logits`）の中で、最後のトークンに対応する予測分布を取り出す。
   - ソフトマックス関数（`F.softmax`）によって確率に変換する。
   - `torch.multinomial` により確率分布に基づいたサンプリングを行い、新しい一つの単語（トークンID）を決定する。
   - 決定された単語を現在のシーケンスの末尾に連結(`torch.cat`)し、このステップを `max_new_tokens` の数だけ繰り返す。

```python
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # 1. コンテキスト長（block_size）を超えないように直近のトークンを見る
            idx_cond = idx[:, -self.cfg.block_size :]
            # 2. モデルの予測を実行
            logits, _ = self(idx_cond)
            # 3. リストの一番最後にある次の単語の予測（確率分布）を抽出する
            probs = F.softmax(logits[:, -1, :], dim=-1)
            # 4. multinomialに従ってランダム性を持たせたサンプリングを実行
            next_token = torch.multinomial(probs, num_samples=1)
            # 5. 現在のシーケンスの末尾に連結してループを継続
            idx = torch.cat((idx, next_token), dim=1)
        return idx
```
4. **デコードと出力**: 最後に予測されたすべてのIDの連なりを `tokenizer.decode()` を介して人間が読める文字列に戻し、画面に表示させる。
