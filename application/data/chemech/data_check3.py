# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_check3.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/23 13:03
# https://chat.deepseek.com/a/chat/s/0086ce73-109e-4cba-a329-c458b4c46994

# !/usr/bin/env python3
"""
检查两个USPTO50k数据集目录中文件的重合情况
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys


def load_json_ids(file_path, desc="Loading"):
    """加载JSON文件并提取所有id"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"错误: {file_path} 的内容不是列表格式")
            return set()

        ids = set()
        for item in tqdm(data, desc=desc, leave=False):
            if isinstance(item, dict) and 'id' in item:
                ids.add(item['id'])

        return ids
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return set()
    except json.JSONDecodeError as e:
        print(f"错误: 解析 {file_path} 时出错: {e}")
        return set()
    except Exception as e:
        print(f"错误: 读取 {file_path} 时发生未知错误: {e}")
        return set()


def compare_datasets(data_dir1, data_dir2, data_file_pattern, splits):
    """比较两个数据集目录的重合情况"""

    results = {}

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"处理 split: {split}")
        print(f"{'=' * 60}")

        # 构建文件路径
        file_pattern = data_file_pattern.format(split=split)
        file1 = Path(data_dir1) / file_pattern
        file2 = Path(data_dir2) / file_pattern

        print(f"文件1: {file1}")
        print(f"文件2: {file2}")

        # 加载两个文件的id
        ids1 = load_json_ids(file1, desc=f"加载 {file1.name}")
        ids2 = load_json_ids(file2, desc=f"加载 {file2.name}")

        # 统计信息
        count1 = len(ids1)
        count2 = len(ids2)
        intersection = ids1 & ids2  # 交集
        union = ids1 | ids2  # 并集
        only_in_1 = ids1 - ids2  # 只在文件1中
        only_in_2 = ids2 - ids1  # 只在文件2中

        intersection_count = len(intersection)
        union_count = len(union)
        only_in_1_count = len(only_in_1)
        only_in_2_count = len(only_in_2)

        # 计算重合率
        overlap_rate_file1 = intersection_count / count1 * 100 if count1 > 0 else 0
        overlap_rate_file2 = intersection_count / count2 * 100 if count2 > 0 else 0

        print(f"\n{split.upper()} 集统计结果:")
        print(f"文件1数据条数: {count1}")
        print(f"文件2数据条数: {count2}")
        print(f"重合数据条数: {intersection_count}")
        print(f"文件1重合率: {overlap_rate_file1:.2f}%")
        print(f"文件2重合率: {overlap_rate_file2:.2f}%")
        print(f"只在文件1中的数据: {only_in_1_count} 条")
        print(f"只在文件2中的数据: {only_in_2_count} 条")

        # 存储结果
        results[split] = {
            'file1_path': str(file1),
            'file2_path': str(file2),
            'file1_count': count1,
            'file2_count': count2,
            'intersection_count': intersection_count,
            'only_in_1_count': only_in_1_count,
            'only_in_2_count': only_in_2_count,
            'overlap_rate_file1': overlap_rate_file1,
            'overlap_rate_file2': overlap_rate_file2,
            'intersection_ids': list(intersection),
            'only_in_1_ids': list(only_in_1),
            'only_in_2_ids': list(only_in_2)
        }

    return results


def check_cross_split_overlap(results, splits):
    """检查不同split之间的重合情况"""
    print(f"\n{'=' * 60}")
    print("检查不同split之间的重合情况")
    print(f"{'=' * 60}")

    all_ids = {}
    for split in splits:
        if split in results:
            # 合并每个split中两个文件的所有id
            file1_ids = set(results[split]['intersection_ids']) | set(results[split]['only_in_1_ids'])
            file2_ids = set(results[split]['intersection_ids']) | set(results[split]['only_in_2_ids'])
            all_ids[f"{split}_file1"] = file1_ids
            all_ids[f"{split}_file2"] = file2_ids

    # 检查不同split之间的重合
    split_combinations = []
    for i, split1 in enumerate(splits):
        for j, split2 in enumerate(splits):
            if i < j:  # 避免重复比较
                split_combinations.append((split1, split2))

    cross_overlap_results = {}

    for split1, split2 in split_combinations:
        print(f"\n检查 {split1} 和 {split2} 之间的重合:")

        for file_type in ['file1', 'file2']:
            key1 = f"{split1}_{file_type}"
            key2 = f"{split2}_{file_type}"

            if key1 in all_ids and key2 in all_ids:
                ids1 = all_ids[key1]
                ids2 = all_ids[key2]
                intersection = ids1 & ids2
                intersection_count = len(intersection)

                total_unique = len(ids1 | ids2)
                overlap_rate = intersection_count / total_unique * 100 if total_unique > 0 else 0

                print(f"  {file_type}: 重合 {intersection_count} 条数据, 重合率: {overlap_rate:.2f}%")

                cross_overlap_results[f"{split1}_{split2}_{file_type}"] = {
                    'intersection_count': intersection_count,
                    'overlap_rate': overlap_rate,
                    'intersection_ids': list(intersection)
                }

    return cross_overlap_results


def main(data_dir1, data_dir2, data_file_pattern, splits):
    """主函数"""
    print("开始比较数据集...")
    print(f"数据目录1: {data_dir1}")
    print(f"数据目录2: {data_dir2}")
    print(f"文件模式: {data_file_pattern}")
    print(f"处理的splits: {splits}")

    # 比较相同split的文件
    results = compare_datasets(data_dir1, data_dir2, data_file_pattern, splits)

    # 检查不同split之间的重合
    cross_results = check_cross_split_overlap(results, splits)

    # 汇总结果
    print(f"\n{'=' * 60}")
    print("汇总结果")
    print(f"{'=' * 60}")

    total_intersection = sum(results[split]['intersection_count'] for split in splits if split in results)
    total_file1 = sum(results[split]['file1_count'] for split in splits if split in results)
    total_file2 = sum(results[split]['file2_count'] for split in splits if split in results)

    overall_overlap_rate1 = total_intersection / total_file1 * 100 if total_file1 > 0 else 0
    overall_overlap_rate2 = total_intersection / total_file2 * 100 if total_file2 > 0 else 0

    print(f"总数据条数 - 文件1: {total_file1}")
    print(f"总数据条数 - 文件2: {total_file2}")
    print(f"总重合数据条数: {total_intersection}")
    print(f"总体重合率 (相对于文件1): {overall_overlap_rate1:.2f}%")
    print(f"总体重合率 (相对于文件2): {overall_overlap_rate2:.2f}%")

    return results, cross_results


if __name__ == "__main__":
    # 设置默认参数
    DEFAULT_DATA_DIR1 = "/mnt/e/DataSets/Chemistry/USPTO50k_RAW/processed"
    DEFAULT_DATA_DIR2 = "/mnt/e/DataSets/Chemistry/USPTO50k/processed"
    DEFAULT_DATA_FILE_PATTERN = "uspto50k_{split}.json"
    DEFAULT_SPLITS = ["train", "test", "val"]

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="比较两个USPTO50k数据集目录的重合情况")
    parser.add_argument("--data_dir1", type=str, default=DEFAULT_DATA_DIR1,
                        help=f"第一个数据目录路径 (默认: {DEFAULT_DATA_DIR1})")
    parser.add_argument("--data_dir2", type=str, default=DEFAULT_DATA_DIR2,
                        help=f"第二个数据目录路径 (默认: {DEFAULT_DATA_DIR2})")
    parser.add_argument("--data_file_pattern", type=str, default=DEFAULT_DATA_FILE_PATTERN,
                        help=f"数据文件命名模式 (默认: {DEFAULT_DATA_FILE_PATTERN})")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                        help=f"要处理的split列表 (默认: {DEFAULT_SPLITS})")

    args = parser.parse_args()

    # 执行主函数
    try:
        results, cross_results = main(
            data_dir1=args.data_dir1,
            data_dir2=args.data_dir2,
            data_file_pattern=args.data_file_pattern,
            splits=args.splits
        )
        print("\n处理完成!")
    except KeyboardInterrupt:
        print("\n用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        sys.exit(1)
