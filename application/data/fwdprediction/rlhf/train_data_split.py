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


def split_dataset(data_file, output_dir, sft_ratio=0.8, reward_ratio=0.1, ppo_ratio=0.1, seed=42):
    """
    将JSON数据集随机切分为sft、reward和ppo三个部分

    Args:
        data_file (str): 输入JSON文件路径
        output_dir (str): 输出目录
        sft_ratio (float): sft set比例
        reward_ratio (float): reward set比例
        ppo_ratio (float): ppo set比例
        seed (int): 随机种子
    """
    # 验证比例总和为1
    total_ratio = sft_ratio + reward_ratio + ppo_ratio
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
    sft_size = int(total_size * sft_ratio)
    reward_size = int(total_size * reward_ratio)

    # 划分数据集
    sft_data = data[:sft_size]
    reward_data = data[sft_size:sft_size + reward_size]
    ppo_data = data[sft_size + reward_size:]

    # 创建输出目录
    splits = ['sft', 'reward', 'ppo']
    for split in splits:
        os.makedirs(os.path.join(output_dir.format(split=split)), exist_ok=True)

    # 获取原始文件名（不含路径）
    base_filename = os.path.basename(data_file)

    # 保存各数据集
    output_files = {
        'sft': os.path.join(output_dir.format(split='sft'), base_filename),
        'reward': os.path.join(output_dir.format(split='reward'), base_filename),
        'ppo': os.path.join(output_dir.format(split='ppo'), base_filename)
    }

    with open(output_files['sft'], 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)

    with open(output_files['reward'], 'w', encoding='utf-8') as f:
        json.dump(reward_data, f, indent=2, ensure_ascii=False)

    with open(output_files['ppo'], 'w', encoding='utf-8') as f:
        json.dump(ppo_data, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print(f"数据集划分完成:")
    print(f"总样本数: {total_size}")
    print(f"sft set: {len(sft_data)} 样本 ({len(sft_data) / total_size * 100:.1f}%)")
    print(f"reward set: {len(reward_data)} 样本 ({len(reward_data) / total_size * 100:.1f}%)")
    print(f"ppo set: {len(ppo_data)} 样本 ({len(ppo_data) / total_size * 100:.1f}%)")
    print(f"输出文件:")
    print(f"  sft set: {output_files['sft']}")
    print(f"  reward set: {output_files['reward']}")
    print(f"  ppo set: {output_files['ppo']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机切分JSON数据集")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="输入文件所在目录")
    parser.add_argument("--data_file", type=str, required=True,
                        help="输入JSON文件名")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录模板，使用{split}占位符")
    parser.add_argument("--sft_ratio", type=float, default=0.4,
                        help="sft set比例")
    parser.add_argument("--reward_ratio", type=float, default=0.3,
                        help="reward set比例")
    parser.add_argument("--ppo_ratio", type=float, default=0.3,
                        help="ppo set比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 构建完整文件路径
    input_file = os.path.join(args.base_dir, args.data_file)

    # 执行数据集切分
    split_dataset(
        data_file=input_file,
        output_dir=args.output_dir,
        sft_ratio=args.sft_ratio,
        reward_ratio=args.reward_ratio,
        ppo_ratio=args.ppo_ratio,
        seed=args.seed
    )
