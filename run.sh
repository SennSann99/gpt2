#!/usr/bin/env bash
set -e

# デフォルト設定
IMAGE_NAME="gpt2"
SESSION_NAME="gpt2"
USE_GPU=false
USE_TMUX=false

# 引数の解析
EXTRA_ARGS=()
for arg in "$@"; do
    case $arg in
        gpu)  USE_GPU=true ;;
        tmux) USE_TMUX=true ;;
        *)    EXTRA_ARGS+=("$arg") ;;
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

# 実行コマンドの構築 (配列で管理)
DOCKER_CMD=(
    docker run --rm --init -it $GPU_OPT
    -v "$PROJECT_ROOT/logs:/workspace/logs"
    -v "$PROJECT_ROOT/checkpoints:/workspace/checkpoints"
    -v "$PROJECT_ROOT/data:/workspace/data"
    -e PYTHONUNBUFFERED=1
    $IMAGE_NAME
)

# 引数が渡されている場合の処理
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    # 最初の引数が "-" で始まる（フラグである）場合、デフォルトの実行コマンドを補完する
    if [[ "${EXTRA_ARGS[0]}" == -* ]]; then
        DOCKER_CMD+=(uv run main.py "${EXTRA_ARGS[@]}")
    else
        # bash など、直接コマンドが指定された場合はそのまま渡す
        DOCKER_CMD+=("${EXTRA_ARGS[@]}")
    fi
fi

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
        # tmux new-session にはコマンドを1つの引数（文字列）として渡す
        tmux new-session -s "$SESSION_NAME" "${DOCKER_CMD[*]}"
    fi
else
    echo "🐳 Running container..."
    # exec を使うことでシグナルが直接 Docker に届くようにする
    exec "${DOCKER_CMD[@]}"
fi
