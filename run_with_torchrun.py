# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: run_with_torchrun.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/6/5 19:26

import subprocess
import os
import sys

def main():
    # 1. 设置环境变量（可选，如果 CUDA_HOME 等需要手动指定）
    os.environ["CUDA_HOME"] = "/home/liangtao/Library/CUDA/cuda-11.8"  # 替换为你的 CUDA 路径
    os.environ["FORCE_TORCHRUN"] = "1"  # 防止某些框架的检查报错

    # 2. 定义 torchrun 命令参数
    torchrun_cmd = [
        "torchrun",
        "--nproc_per_node=1",  # 使用 1 个 GPU（多卡可改为 2、4 等）
        "src/train.py",        # 替换为你的训练脚本路径
        "examples/train_lora/qwen205/demo.yaml"  # 替换为你的配置文件
    ]

    # 3. 打印执行的命令（调试用）
    print("[Running Command]:", " ".join(torchrun_cmd))

    # 4. 调用 torchrun
    try:
        subprocess.run(torchrun_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()