# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_convert_split.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/19 22:39
# https://chat.deepseek.com/a/chat/s/2ea69a31-19e2-4180-8e27-8e94325a6334

import json
import os
import random
from tqdm import tqdm
import argparse


def process_chemical_mechanism_data(data_dir, data_file, output_dir, splits, ratios, seed):
    """
    处理化学机理数据并分割为训练集、验证集和测试集

    Args:
        data_dir: 原始数据目录
        data_file: 原始数据文件名
        output_dir: 输出目录
        splits: 分割名称列表
        ratios: 分割比例
        seed: 随机种子
    """
    # 设置随机种子以确保可重复性
    random.seed(seed)

    # 构建完整文件路径
    input_file_path = os.path.join(data_dir, "processed", data_file)
    output_dir_path = os.path.join(data_dir, output_dir)

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir_path, exist_ok=True)

    print(f"正在读取数据文件: {input_file_path}")

    # 读取原始数据
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    print(f"成功读取 {len(data)} 条数据")

    # 随机打乱数据
    random.shuffle(data)

    # 计算各分割的数据量
    total_size = len(data)
    train_size = int(total_size * ratios[0])
    valid_size = int(total_size * ratios[1])
    test_size = total_size - train_size - valid_size

    print(f"数据分割: 训练集 {train_size} 条, 验证集 {valid_size} 条, 测试集 {test_size} 条")

    # 分割数据
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]

    # 处理每个分割的数据
    split_data = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }

    for split in splits:
        print(f"\n正在处理 {split} 数据集...")
        output_data = []

        # 为每个样本创建进度条
        for i, item in enumerate(tqdm(split_data[split], desc=f"处理 {split} 数据")):
            try:
                output_item = {
                    "id": f"chemech_prediction_{split}_{i:09d}",
                    "instruction": "PREDICT_CHEMECH:",
                    "input": item["updated_reaction"],
                    "output": item["mechanistic_label"]
                }
                output_data.append(output_item)
            except KeyError as e:
                print(f"跳过数据项 {i}: 缺少键 {e}")
                continue

        # 构建输出文件路径
        output_file = os.path.join(output_dir_path, f"{split}.json")

        # 写入输出文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"成功写入 {len(output_data)} 条数据到 {output_file}")
        except Exception as e:
            print(f"写入文件时出错: {e}")

    print("\n数据处理完成！")


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="处理化学机理数据并分割为训练集、验证集和测试集")

    parser.add_argument('--data_dir', type=str, default='/mnt/e/DataSets/Chemistry/ChemicalMechanism/',
                        help='原始数据目录')
    parser.add_argument('--data_file', type=str, default='mech-USPTO-31k.json',
                        help='原始数据文件名')
    parser.add_argument('--output_dir', type=str, default='chemech_prediction',
                        help='输出目录')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='分割名称列表')
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.8, 0.1, 0.1],
                        help='分割比例（训练集、验证集、测试集）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 确保比例总和为1
    if sum(args.ratios) != 1.0:
        print("警告：比例总和不为1，将自动归一化")
        total = sum(args.ratios)
        args.ratios = [r / total for r in args.ratios]

    # 调用处理函数
    process_chemical_mechanism_data(
        data_dir=args.data_dir,
        data_file=args.data_file,
        output_dir=args.output_dir,
        splits=args.splits,
        ratios=args.ratios,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
