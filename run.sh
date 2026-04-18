#!/usr/bin/env bash
set -e

# デフォルト設定
IMAGE_NAME="gpt2"
SESSION_NAME="gpt2"
USE_GPU=false
USE_TMUX=false
USE_WANDB=false

# 引数の解析
EXTRA_ARGS=()
for arg in "$@"; do
    case $arg in
        gpu)   USE_GPU=true ;;
        tmux)  USE_TMUX=true ;;
        wandb) USE_WANDB=true ;;
        *)     EXTRA_ARGS+=("$arg") ;;
    esac
done

# プロジェクトルートの絶対パスを取得
PROJECT_ROOT=$(pwd)

# 必要なディレクトリの作成
mkdir -p logs checkpoints data

# イメージのビルド
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# GPUオプションの構築
GPU_OPT=()
if [ "$USE_GPU" = true ]; then
    GPU_OPT=("--gpus" "all")
    echo "GPU support enabled."
fi

# 実行コマンドの構築 (配列で管理)
DOCKER_CMD=(
    docker run --rm --init -it "${GPU_OPT[@]}"
    -v "$PROJECT_ROOT/logs:/workspace/logs"
    -v "$PROJECT_ROOT/checkpoints:/workspace/checkpoints"
    -v "$PROJECT_ROOT/data:/workspace/data"
    -e PYTHONUNBUFFERED=1
)

# WANDBオプションの処理
WANDB_FLAG=()
if [ "$USE_WANDB" = true ]; then
    if [ -f .env ]; then
        # .env から WANDB_API_KEY を抽出して環境変数として渡す
        WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d '=' -f2 | xargs)
        if [ -n "$WANDB_API_KEY" ]; then
            DOCKER_CMD+=("-e" "WANDB_API_KEY=$WANDB_API_KEY")
            WANDB_FLAG=("--wandb")
            echo "WandB logging enabled."
        else
            echo "Warning: WANDB_API_KEY not found in .env. Logging might fail."
        fi
    else
        echo "Warning: .env file not found. Logging might fail."
    fi
fi

# イメージ名の追加
DOCKER_CMD+=("$IMAGE_NAME")

# 引数が渡されている場合の処理
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    # 最初の引数が "-" で始まる（フラグである）場合
    if [[ "${EXTRA_ARGS[0]}" == -* ]]; then
        DOCKER_CMD+=(uv run main.py "${WANDB_FLAG[@]}" "${EXTRA_ARGS[@]}")
    else
        # bash など、直接コマンドが指定された場合はそのまま渡す（main.py は実行しない）
        DOCKER_CMD+=("${EXTRA_ARGS[@]}")
    fi
else
    # 追加引数がない場合はデフォルトの main.py を実行
    DOCKER_CMD+=(uv run main.py "${WANDB_FLAG[@]}")
fi

if [ "$USE_TMUX" = true ]; then
    if ! command -v tmux > /dev/null; then
        echo "Error: tmux is not installed."
        exit 1
    fi

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session '$SESSION_NAME' already exists. Attaching..."
        tmux attach-session -t "$SESSION_NAME"
    else
        echo "Starting new tmux session '$SESSION_NAME'..."
        # tmux new-session にはコマンドを1つの文字列として渡す必要があるため [*] を使用
        tmux new-session -s "$SESSION_NAME" "${DOCKER_CMD[*]}"
    fi
else
    echo "Running container..."
    # exec を使うことでシグナルが直接 Docker に届くようにする
    exec "${DOCKER_CMD[@]}"
fi