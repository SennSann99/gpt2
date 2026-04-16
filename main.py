import sys

from gpt2.generate import main as generate_main
from gpt2.train import main as train_main

if __name__ == "__main__":
    try:
        train_main()
    except KeyboardInterrupt:
        # 学習中断時は生成に進まずに終了
        sys.exit(0)

    generate_main()
