# Data Pipeline (データパイプライン)

学習データには、小規模モデル向けのマルチターン会話データセット [`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) を使用する。各レコードは`messages`列を持ち、`system`、`user`、`assistant`のロールで会話履歴を表現する。

## 全体の流れ

1. Hugging Face Datasetsから`train`と`test`を読み込む
2. 各メッセージを学習用チャット形式へ変換する
3. GPT-2トークナイザで会話全体をトークン化する
4. 直近の`block_size`トークンを残し、不足分をパディングする
5. Assistant以外のラベルを`-100`にし、損失計算から除外する
6. PyTorch `DataLoader`でLightningへ供給する

## チャット形式

```text
System: You are a helpful assistant.
User: What is RoPE?
Model: RoPE is a rotary position encoding method.<|endoftext|>
User: Why is it useful?
Model:
```

学習時は会話履歴全体を入力として使用するが、損失はAssistantの応答トークンに対してのみ計算する。これにより、モデルは過去のやり取りを参照しながら次の応答を学習できる。

## コンテキスト長

デフォルトの`block_size`は`512`である。会話が512トークンを超える場合は、最新のAssistant応答が含まれるように直近の履歴を保存する。推論時もStreamlit UIが保持する会話履歴の末尾512トークンをモデルへ渡す。

## データ量の制限

`--limit-rows`に正の値を指定すると、学習スプリットの先頭から使用する会話数を制限できる。`0`は全件を意味する。`--val-rows`は検証スプリットに使用する会話数を指定する。
