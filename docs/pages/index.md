# GPT-2 Implementation Documentation

このディレクトリ（`/docs`）には、PyTorch Lightningベースの簡易GPT-2実装に関するアーキテクチャや学習系の詳細なドキュメントが保管されている。この実装は、研究目的や学習のしやすさ、そして拡張性を重視して構築されたものである。

## 目次

1. [**アーキテクチャとモデル (architecture.md)**](architecture.md)
   - `model.py` の詳細解説
   - GPTModelの基本構成要素
   - Rotary Position Embedding (RoPE) の仕組み
   - Causal Self-Attentionの詳細

2. [**データパイプライン (data_pipeline.md)**](data_pipeline.md)
   - Smol-SmolTalkのマルチターン会話の読み込み
   - GPT-2トークナイザによる会話履歴のトークン化
   - Assistant応答のみを学習対象とするラベル作成
   - PyTorch `DataLoader`の構築

3. [**学習と生成ループ (training_and_generation.md)**](training_and_generation.md)
   - `train.py` および `generate.py` の詳細解説
   - PyTorch Lightningの `Trainer` を使った学習の仕組み
   - multinomialサンプリングによるテキストの生成手法

4. [**設定とパラメータ (configuration.md)**](configuration.md)
   - `config.py` の詳細解説
   - `ModelConfig` および `TrainConfig` それぞれのパラメータ一覧
   - `@dataclass(slots=True)` による最適化機構

各ドキュメントを通じて、このプロジェクトの意図と実装の裏にあるアイデアを網羅的に確認できる。学習・研究したい部分のリンクを参照されたい。
