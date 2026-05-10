#!/usr/bin/env bash
set -e

# 設定
IMAGE_NAME="gpt2-app"
PORT=8501

# プロジェクトルートの絶対パスを取得（このスクリプトはfrontend/にあるが、ルートでビルドする）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 必要なディレクトリの作成
mkdir -p "$PROJECT_ROOT/checkpoints"

# GPU オプションの解析
USE_GPU=false
for arg in "$@"; do
    case $arg in
        gpu) USE_GPU=true ;;
    esac
done

GPU_OPT=()
if [ "$USE_GPU" = true ]; then
    GPU_OPT=("--gpus" "all")
    echo "GPU support enabled."
fi

# イメージのビルド（プロジェクトルートをコンテキストにする）
echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"

# コンテナの起動
echo "Starting Streamlit app on http://localhost:$PORT ..."
exec docker run --rm -it "${GPU_OPT[@]}" \
    -p "$PORT:8501" \
    -v "$PROJECT_ROOT/checkpoints:/workspace/checkpoints" \
    "$IMAGE_NAME"
