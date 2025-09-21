# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_split.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/12 9:39
# https://chat.deepseek.com/a/chat/s/2e19af52-0fd0-4ea1-bba0-d5ba28f30989

import json
import os
import random
import argparse
from typing import List, Dict, Any


def split_dataset(data: List[Dict[str, Any]], split_ratio: float = 0.5, seed: int = 42) -> tuple:
    """
    随机切分数据集为两个部分

    Args:
        data: 输入的数据列表
        split_ratio: 第一个split的比例
        seed: 随机种子

    Returns:
        tuple: (split1, split2) 两个切分后的数据集
    """
    # 设置随机种子以确保可重复性
    random.seed(seed)

    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 计算切分点
    split_point = int(len(shuffled_data) * split_ratio)

    # 切分数据
    split1 = shuffled_data[:split_point]
    split2 = shuffled_data[split_point:]

    return split1, split2


def process_dataset(base_dir: str, data_file: str, output_dir: str, split_ratio: float = 0.5, seed: int = 42):
    """
    处理数据集并保存切分结果

    Args:
        base_dir: 输入文件目录
        data_file: 输入文件名
        output_dir: 输出目录模板（包含{split}占位符）
        split_ratio: 切分比例
        seed: 随机种子
    """
    # 构建完整的输入文件路径
    input_path = os.path.join(base_dir, data_file)

    # 读取JSON文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"成功读取文件: {input_path}")
        print(f"数据总数: {len(data)}")

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 切分数据集
    split1, split2 = split_dataset(data, split_ratio, seed)

    print(f"Split1 大小: {len(split1)}")
    print(f"Split2 大小: {len(split2)}")

    # 创建输出目录
    split1_dir = output_dir.format(split="split1")
    split2_dir = output_dir.format(split="split2")

    os.makedirs(split1_dir, exist_ok=True)
    os.makedirs(split2_dir, exist_ok=True)

    # 构建输出文件路径
    output_path1 = os.path.join(split1_dir, data_file)
    output_path2 = os.path.join(split2_dir, data_file)

    # 保存切分后的数据
    try:
        with open(output_path1, 'w', encoding='utf-8') as f:
            json.dump(split1, f, indent=2, ensure_ascii=False)

        with open(output_path2, 'w', encoding='utf-8') as f:
            json.dump(split2, f, indent=2, ensure_ascii=False)

        print(f"Split1 已保存到: {output_path1}")
        print(f"Split2 已保存到: {output_path2}")

    except Exception as e:
        print(f"保存文件时出错: {e}")


def main():

    # 执行数据处理
    process_dataset(
        base_dir=args.base_dir,
        data_file=args.data_file,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    """主函数，处理命令行参数并执行切分"""
    parser = argparse.ArgumentParser(description="随机切分JSON数据集")

    parser.add_argument("--base_dir", type=str, default="/mnt/e/DataSets/Chemistry/RetroSynthesis",
                        help="输入文件所在的基础目录")
    parser.add_argument("--data_file", type=str, default="retrosynthesis_train.json",
                        help="输入JSON文件名")
    parser.add_argument("--output_dir", type=str, default="/mnt/e/DataSets/Chemistry/RetroSynthesis/v1/{split}/",
                        help="输出目录模板，包含{split}占位符")
    parser.add_argument("--split_ratio", type=float, default=0.5,
                        help="第一个split的比例，默认为0.5")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，默认为42")

    args = parser.parse_args()
    main()
