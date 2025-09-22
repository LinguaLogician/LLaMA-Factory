# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_merge_mech.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/22 22:15
# https://chat.deepseek.com/a/chat/s/345e9279-56e0-481e-8d81-6355f2d98e7f
import os
import json
import argparse
from tqdm import tqdm
from glob import glob
from typing import Dict, List


def merge_chemistry_data(data_dir: str = "/mnt/e/DataSets/Chemistry/Chemech/v1",
                         data_file_pattern: str = "mech-USPTO_{split}.json",
                         target_file_pattern: str = "mech-USPTO_{split}_merged.json",
                         splits: List[str] = None):
    """
    合并data_dir下各个文件夹中对应split的json文件

    Args:
        data_dir: 数据目录路径
        data_file_pattern: 数据文件名模式，包含{split}占位符
        target_file_pattern: 目标文件名模式，包含{split}占位符
        splits: 要处理的split列表，默认为['train', 'test', 'val']
    """

    if splits is None:
        splits = ['train', 'test', 'val']

    # 获取data_dir下的所有子文件夹
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]

    print(f"在目录 {data_dir} 下找到 {len(subdirs)} 个子文件夹")
    print(f"开始处理 splits: {splits}")
    print("-" * 50)

    # 为每个split创建合并后的数据列表
    merged_data = {split: [] for split in splits}
    file_counts = {split: 0 for split in splits}

    # 遍历每个子文件夹
    for subdir in tqdm(subdirs, desc="处理子文件夹"):
        subdir_path = os.path.join(data_dir, subdir)

        # 检查每个split对应的文件是否存在
        for split in splits:
            data_file_path = os.path.join(subdir_path, data_file_pattern.format(split=split))

            if os.path.exists(data_file_path):
                try:
                    # 读取JSON文件
                    with open(data_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 确保数据是列表格式
                    if isinstance(data, list):
                        merged_data[split].extend(data)
                        file_counts[split] += 1
                    else:
                        print(f"警告: {data_file_path} 中的数据不是列表格式，跳过")

                except json.JSONDecodeError as e:
                    print(f"错误: 无法解析JSON文件 {data_file_path}: {e}")
                except Exception as e:
                    print(f"错误: 处理文件 {data_file_path} 时发生异常: {e}")

    print("-" * 50)
    print("文件处理统计:")
    for split in splits:
        print(f"  {split}: 处理了 {file_counts[split]} 个文件，合并了 {len(merged_data[split])} 条数据")

    # 保存合并后的文件
    print("-" * 50)
    print("开始保存合并后的文件...")

    for split in splits:
        if merged_data[split]:  # 只保存有数据的split
            target_file = target_file_pattern.format(split=split)
            target_path = os.path.join(data_dir, target_file)

            try:
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_data[split], f, ensure_ascii=False, indent=2)

                print(f"成功保存: {target_path} (包含 {len(merged_data[split])} 条数据)")

            except Exception as e:
                print(f"错误: 保存文件 {target_path} 时发生异常: {e}")
        else:
            print(f"警告: split '{split}' 没有数据，跳过保存")

    print("处理完成!")


def main():
    """主函数，处理命令行参数并执行合并操作"""
    parser = argparse.ArgumentParser(description='合并化学机制数据文件')
    parser.add_argument('--data_dir', type=str,
                        default="/mnt/e/DataSets/Chemistry/Chemech2",
                        help='数据目录路径')
    parser.add_argument('--data_file_pattern', type=str,
                        default="mech-USPTO_{split}_merged.json",
                        help='数据文件名模式，包含{split}占位符')
    parser.add_argument('--target_file_pattern', type=str,
                        default="mech-USPTO_{split}_merged.json",
                        help='目标文件名模式，包含{split}占位符')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'test', 'val'],
                        help='要处理的split列表，用空格分隔')

    args = parser.parse_args()

    # 执行合并操作
    merge_chemistry_data(
        data_dir=args.data_dir,
        data_file_pattern=args.data_file_pattern,
        target_file_pattern=args.target_file_pattern,
        splits=args.splits
    )


if __name__ == "__main__":
    main()
