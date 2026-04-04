#!/usr/bin/env bash
set -e

# デフォルト設定
IMAGE_NAME="gpt2"
SESSION_NAME="gpt2"
USE_GPU=false
USE_TMUX=false

# 引数の解析
for arg in "$@"; do
    case $arg in
        gpu)  USE_GPU=true ;;
        tmux) USE_TMUX=true ;;
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

# 実行コマンドの構築
RUN_CMD="docker run --rm --init -it $GPU_OPT \
    -v \"$PROJECT_ROOT/logs:/workspace/logs\" \
    -v \"$PROJECT_ROOT/checkpoints:/workspace/checkpoints\" \
    -v \"$PROJECT_ROOT/data:/workspace/data\" \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE_NAME"

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
        # tmux の場合はセッション維持のため exec ではなく普通に実行
        tmux new-session -s "$SESSION_NAME" "$RUN_CMD"
    fi
else
    echo "🐳 Running container..."
    # exec を使うことでシグナルが直接 Docker に届くようにする
    exec docker run --rm --init -it $GPU_OPT \
        -v "$PROJECT_ROOT/logs:/workspace/logs" \
        -v "$PROJECT_ROOT/checkpoints:/workspace/checkpoints" \
        -v "$PROJECT_ROOT/data:/workspace/data" \
        -e PYTHONUNBUFFERED=1 \
        $IMAGE_NAME
fi
