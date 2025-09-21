# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_split.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/20 15:55
# https://chat.deepseek.com/a/chat/s/e13569cd-8eae-4e78-a71f-d2216598e8a7
import json
import os
import random
import argparse
from tqdm import tqdm
from typing import Tuple, Dict, List


def split_dataset(
        data_dir: str = "/mnt/e/DataSets/Chemistry/ChemicalMechanism",
        data_file: str = "mech-USPTO-31k.json",
        output_dir: str = "/mnt/e/DataSets/Chemistry/ChemicalMechanism",
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
) -> None:
    """
    将数据集分割为训练集、验证集和测试集

    Args:
        data_dir: 输入数据目录
        data_file: 输入数据文件名
        output_dir: 输出根目录
        ratios: 分割比例 (train, valid, test)
        seed: 随机种子
    """
    # 设置随机种子以确保可重复性
    random.seed(seed)

    # 检查比例总和是否为1
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为1，当前总和为: {sum(ratios)}")

    # 构建完整的数据文件路径
    input_path = os.path.join(data_dir, data_file)

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    print(f"正在读取数据文件: {input_path}")

    # 读取JSON数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"成功读取 {len(data)} 条数据")

    # 随机打乱数据
    print("正在随机打乱数据...")
    random.shuffle(data)

    # 计算各分割的大小
    total_size = len(data)
    train_size = int(total_size * ratios[0])
    valid_size = int(total_size * ratios[1])
    test_size = total_size - train_size - valid_size

    print(f"分割大小 - 训练集: {train_size}, 验证集: {valid_size}, 测试集: {test_size}")

    # 分割数据
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]

    # 定义分割名称和对应的数据
    splits = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }

    # 处理每个分割
    for split_name, split_data in splits.items():
        # 创建输出目录（如果不存在）
        split_output_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)

        # 构建输出文件名
        output_filename = f"{os.path.splitext(data_file)[0]}_{split_name}.json"
        output_path = os.path.join(split_output_dir, output_filename)

        print(f"正在写入 {split_name} 分割数据到: {output_path}")

        # 写入JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        print(f"成功写入 {len(split_data)} 条数据到 {split_name} 分割")

    print("数据集分割完成！")


def main():
    """主函数，处理命令行参数并执行分割"""
    parser = argparse.ArgumentParser(description="分割化学机制数据集")

    parser.add_argument("--data_dir", type=str,
                        default="/mnt/e/DataSets/Chemistry/ChemicalMechanism",
                        help="输入数据目录")
    parser.add_argument("--data_file", type=str,
                        default="mech-USPTO-31k.json",
                        help="输入数据文件名")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/e/DataSets/Chemistry/ChemicalMechanism",
                        help="输出根目录")
    parser.add_argument("--train_ratio", type=float,
                        default=0.8, help="训练集比例")
    parser.add_argument("--valid_ratio", type=float,
                        default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float,
                        default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int,
                        default=42, help="随机种子")

    args = parser.parse_args()

    # 调用分割函数
    split_dataset(
        data_dir=args.data_dir,
        data_file=args.data_file,
        output_dir=args.output_dir,
        ratios=(args.train_ratio, args.valid_ratio, args.test_ratio),
        seed=args.seed
    )


if __name__ == "__main__":
    main()
