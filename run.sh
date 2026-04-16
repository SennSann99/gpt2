#!/usr/bin/env bash
set -e

# デフォルト設定
IMAGE_NAME="gpt2"
SESSION_NAME="gpt2"
USE_GPU=false
USE_TMUX=false

# 引数の解析
USE_WANDB=false
for arg in "$@"; do
    case $arg in
        gpu)   USE_GPU=true ;;
        tmux)  USE_TMUX=true ;;
        wandb) USE_WANDB=true ;;
    esac
done

# プロジェクトルートの絶対パスを取得
PROJECT_ROOT=$(pwd)

# 必要なディレクトリの作成
mkdir -p logs checkpoints data

# イメージのビルド
echo "🚀 Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# GPUオプションの構築
GPU_OPT=""
if [ "$USE_GPU" = true ]; then
    GPU_OPT="--gpus all"
    echo "💡 GPU support enabled."
fi

# WANDBオプションの構築
WANDB_ENV=""
WANDB_OPT=""
if [ "$USE_WANDB" = true ]; then
    if [ -f .env ]; then
        # .env から WANDB_API_KEY を抽出して環境変数として渡す
        WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d '=' -f2 | xargs)
        if [ -n "$WANDB_API_KEY" ]; then
            WANDB_ENV="-e WANDB_API_KEY=$WANDB_API_KEY"
            WANDB_OPT="--wandb"
            echo "📊 WandB logging enabled."
        else
            echo "⚠️  Warning: WANDB_API_KEY not found in .env. Logging might fail."
        fi
    else
        echo "⚠️  Warning: .env file not found. Logging might fail."
    fi
fi

# 実行コマンドの構築
# CMDの末尾に引数を追加できるように $IMAGE_NAME の後に uv run main.py $WANDB_OPT を追加
RUN_CMD="docker run --rm -it $GPU_OPT $WANDB_ENV \
    -v \"$PROJECT_ROOT/logs:/workspace/logs\" \
    -v \"$PROJECT_ROOT/checkpoints:/workspace/checkpoints\" \
    -v \"$PROJECT_ROOT/data:/workspace/data\" \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE_NAME uv run main.py $WANDB_OPT"

if [ "$USE_TMUX" = true ]; then
    if ! command -v tmux > /dev/null; then
        echo "❌ Error: tmux is not installed."
        exit 1
    fi

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️  Session '$SESSION_NAME' already exists. Attaching..."
        tmux attach-session -t "$SESSION_NAME"
    else
        echo "🖥️  Starting new tmux session '$SESSION_NAME'..."
        tmux new-session -s "$SESSION_NAME" "$RUN_CMD"
    fi
else
    echo "🐳 Running container..."
    eval "$RUN_CMD"
fi
