#!/usr/bin/env bash
set -e

IMAGE_NAME="gpt2-app"
PORT=8501
CHECKPOINT_PATH="${1:-checkpoints/version_0/best.ckpt}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building Docker image..."
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"

echo "Starting app with: $CHECKPOINT_PATH"
exec docker run --rm -it --gpus all \
    -p "$PORT:8501" \
    --mount type=bind,src="$PROJECT_ROOT/checkpoints",dst=/workspace/checkpoints,readonly \
    -e "CHECKPOINT_PATH=$CHECKPOINT_PATH" \
    "$IMAGE_NAME"