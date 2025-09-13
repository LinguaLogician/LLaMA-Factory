# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_split.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/12 9:39
# https://chat.deepseek.com/a/chat/s/dda08fba-4c83-4993-9419-d55e8eafb893

import json
import random
import os
import argparse


def split_dataset(data_file, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """
    将JSON数据集随机切分为train、valid和test三个部分

    Args:
        data_file (str): 输入JSON文件路径
        output_dir (str): 输出目录
        train_ratio (float): 训练集比例
        valid_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        seed (int): 随机种子
    """
    # 验证比例总和为1
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和应为1.0，当前为{total_ratio}")

    # 读取原始数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机打乱数据
    random.seed(seed)
    random.shuffle(data)

    # 计算各数据集大小
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)

    # 划分数据集
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]

    # 创建输出目录
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir.format(split=split)), exist_ok=True)

    # 获取原始文件名（不含路径）
    base_filename = os.path.basename(data_file)

    # 保存各数据集
    output_files = {
        'train': os.path.join(output_dir.format(split='train'), base_filename),
        'valid': os.path.join(output_dir.format(split='valid'), base_filename),
        'test': os.path.join(output_dir.format(split='test'), base_filename)
    }

    with open(output_files['train'], 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(output_files['valid'], 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    with open(output_files['test'], 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print(f"数据集划分完成:")
    print(f"总样本数: {total_size}")
    print(f"训练集: {len(train_data)} 样本 ({len(train_data) / total_size * 100:.1f}%)")
    print(f"验证集: {len(valid_data)} 样本 ({len(valid_data) / total_size * 100:.1f}%)")
    print(f"测试集: {len(test_data)} 样本 ({len(test_data) / total_size * 100:.1f}%)")
    print(f"输出文件:")
    print(f"  训练集: {output_files['train']}")
    print(f"  验证集: {output_files['valid']}")
    print(f"  测试集: {output_files['test']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机切分JSON数据集")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="输入文件所在目录")
    parser.add_argument("--data_file", type=str, required=True,
                        help="输入JSON文件名")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录模板，使用{split}占位符")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 构建完整文件路径
    input_file = os.path.join(args.base_dir, args.data_file)

    # 执行数据集切分
    split_dataset(
        data_file=input_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
