# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_check4.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/23 13:13
# https://chat.deepseek.com/a/chat/s/da25d2d8-90b6-4e9a-a2ad-da904222f8ca
# !/usr/bin/env python3
"""
检查USPTO50k数据集中不同split之间的数据重合情况
通过比较数据的id字段来判断是否重合
"""

import json
import os
from pathlib import Path
from typing import Set, Dict, List, Tuple
import argparse
from tqdm import tqdm


def load_data_files(data_dir: str, splits: List[str], data_file_pattern: str = "uspto50k_{split}.json") -> Dict[
    str, List[Dict]]:
    """
    加载指定目录下的数据文件

    Args:
        data_dir: 数据目录路径
        splits: 要加载的split列表
        data_file_pattern: 数据文件名模式

    Returns:
        字典，key为split名称，value为数据列表
    """
    data_dict = {}

    for split in splits:
        file_path = Path(data_dir) / data_file_pattern.format(split=split)

        if not file_path.exists():
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue

        print(f"正在加载 {split} 数据...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data_dict[split] = data
            print(f"{split} 数据加载完成，共 {len(data)} 条记录")

    return data_dict


def extract_ids(data_dict: Dict[str, List[Dict]]) -> Dict[str, Set[str]]:
    """
    从数据中提取id集合

    Args:
        data_dict: 包含数据的字典

    Returns:
        字典，key为split名称，value为id集合
    """
    id_dict = {}

    for split, data_list in data_dict.items():
        print(f"正在提取 {split} 的id...")
        id_set = set()

        for item in tqdm(data_list, desc=f"处理 {split}"):
            if isinstance(item, dict) and 'id' in item:
                id_set.add(item['id'])

        id_dict[split] = id_set
        print(f"{split} 共提取 {len(id_set)} 个唯一id")

    return id_dict


def calculate_overlap(id_dict: Dict[str, Set[str]]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    计算不同split之间的重合情况

    Args:
        id_dict: 包含id集合的字典

    Returns:
        字典，key为split对元组，value为重合统计信息
    """
    splits = list(id_dict.keys())
    overlap_results = {}

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            split1, split2 = splits[i], splits[j]
            set1, set2 = id_dict[split1], id_dict[split2]

            intersection = set1 & set2
            union = set1 | set2

            overlap_info = {
                'split1_count': len(set1),
                'split2_count': len(set2),
                'intersection_count': len(intersection),
                'union_count': len(union),
                'overlap_ratio_split1': len(intersection) / len(set1) if len(set1) > 0 else 0,
                'overlap_ratio_split2': len(intersection) / len(set2) if len(set2) > 0 else 0,
                'jaccard_similarity': len(intersection) / len(union) if len(union) > 0 else 0
            }

            overlap_results[(split1, split2)] = overlap_info

    return overlap_results


def print_overlap_results(overlap_results: Dict[Tuple[str, str], Dict[str, int]]):
    """
    打印重合结果

    Args:
        overlap_results: 重合结果字典
    """
    print("\n" + "=" * 80)
    print("数据重合分析结果")
    print("=" * 80)

    for (split1, split2), stats in overlap_results.items():
        print(f"\n{split1.upper()} vs {split2.upper()}:")
        print(f"  {split1} 数据量: {stats['split1_count']}")
        print(f"  {split2} 数据量: {stats['split2_count']}")
        print(f"  重合数据量: {stats['intersection_count']}")
        print(
            f"  重合比例 ({split1}): {stats['overlap_ratio_split1']:.4f} ({stats['overlap_ratio_split1'] * 100:.2f}%)")
        print(
            f"  重合比例 ({split2}): {stats['overlap_ratio_split2']:.4f} ({stats['overlap_ratio_split2'] * 100:.2f}%)")
        print(f"  Jaccard相似度: {stats['jaccard_similarity']:.4f}")


def analyze_data_overlap(data_dir: str, splits: List[str] = None,
                         data_file_pattern: str = "uspto50k_{split}.json") -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    分析数据重合情况的主函数

    Args:
        data_dir: 数据目录路径
        splits: split列表，默认为['train', 'val', 'test']
        data_file_pattern: 数据文件名模式

    Returns:
        重合分析结果
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    print(f"开始分析数据目录: {data_dir}")
    print(f"处理的splits: {splits}")
    print(f"文件模式: {data_file_pattern}")

    # 加载数据
    data_dict = load_data_files(data_dir, splits, data_file_pattern)

    if not data_dict:
        print("错误: 没有成功加载任何数据文件")
        return {}

    # 提取id
    id_dict = extract_ids(data_dict)

    # 计算重合
    overlap_results = calculate_overlap(id_dict)

    # 打印结果
    print_overlap_results(overlap_results)

    return overlap_results


def main():
    """主函数，处理命令行参数并执行分析"""
    parser = argparse.ArgumentParser(description='分析USPTO50k数据集不同split之间的数据重合情况')
    parser.add_argument('--data_dir', type=str, default='/mnt/e/DataSets/Chemistry/USPTO50k/processed',
                        help='数据目录路径')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='要分析的split名称列表')
    parser.add_argument('--data_file_pattern', type=str, default='uspto50k_{split}.json',
                        help='数据文件名模式，使用{split}作为占位符')

    args = parser.parse_args()

    print("USPTO50k数据集重合分析工具")
    print("=" * 50)

    try:
        analyze_data_overlap(
            data_dir=args.data_dir,
            splits=args.splits,
            data_file_pattern=args.data_file_pattern
        )
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
