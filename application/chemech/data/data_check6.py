# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_check6.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/23 14:12

# https://chat.deepseek.com/a/chat/s/5a3a9f2e-bfba-4e81-a974-c89c002c4aed

# !/usr/bin/env python3
"""
检查RetroSynthesis数据集中不同split之间的数据重合情况
"""

import os
import json
import argparse
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
from collections import defaultdict


def load_data(file_path: str) -> List[Dict]:
    """加载JSON数据文件"""
    print(f"正在加载文件: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"成功加载 {len(data)} 条数据")
    return data


def check_duplicates_within_split(data: List[Dict], split_name: str) -> Dict:
    """
    检查单个split内的重复数据
    返回: 重复数据的统计信息
    """
    print(f"\n正在检查 {split_name} 内部的重复数据...")

    # 使用(input, output)元组作为键来识别重复数据
    data_dict = defaultdict(list)

    for item in tqdm(data, desc=f"处理 {split_name}"):
        key = (item.get('input', ''), item.get('output', ''))
        data_dict[key].append(item.get('id', '未知ID'))

    # 找出重复的数据
    duplicates = {key: ids for key, ids in data_dict.items() if len(ids) > 1}

    print(f"{split_name} 内部重复数据组数: {len(duplicates)}")

    # 统计重复数据详细信息
    duplicate_info = {
        'total_duplicate_groups': len(duplicates),
        'total_duplicate_items': sum(len(ids) for ids in duplicates.values()),
        'duplicate_details': duplicates
    }

    return duplicate_info


def check_overlap_between_splits(data1: List[Dict], data2: List[Dict],
                                 split1_name: str, split2_name: str) -> Dict:
    """
    检查两个split之间的数据重合
    返回: 重合数据的统计信息
    """
    print(f"\n正在检查 {split1_name} 和 {split2_name} 之间的数据重合...")

    # 构建第一个split的数据索引
    split1_data = set()
    split1_id_mapping = {}

    for item in tqdm(data1, desc=f"构建 {split1_name} 索引"):
        key = (item.get('input', ''), item.get('output', ''))
        split1_data.add(key)
        split1_id_mapping[key] = item.get('id', '未知ID')

    # 检查第二个split中的数据是否在第一个split中存在
    overlapping_data = []

    for item in tqdm(data2, desc=f"检查 {split2_name}"):
        key = (item.get('input', ''), item.get('output', ''))
        if key in split1_data:
            overlapping_data.append({
                'id_split1': split1_id_mapping[key],
                'id_split2': item.get('id', '未知ID'),
                'input': key[0],
                'output': key[1]
            })

    print(f"{split1_name} 和 {split2_name} 之间的重合数据条数: {len(overlapping_data)}")

    return {
        'total_overlap': len(overlapping_data),
        'overlapping_items': overlapping_data
    }


def analyze_dataset(data_dir: str, splits: List[str] = None,
                    data_file_pattern: str = "retrosynthesis_{split}.json") -> Dict:
    """
    分析数据集的重合情况

    Args:
        data_dir: 数据目录路径
        splits: 要检查的split列表
        data_file_pattern: 数据文件名模式

    Returns:
        包含所有分析结果的字典
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    print(f"开始分析数据集目录: {data_dir}")
    print(f"检查的splits: {splits}")
    print("=" * 60)

    # 加载所有split的数据
    split_data = {}
    for split in splits:
        file_path = os.path.join(data_dir, f"{data_file_pattern.format(split=split)}")
        split_data[split] = load_data(file_path)

    results = {
        'data_dir': data_dir,
        'splits_analyzed': splits,
        'within_split_duplicates': {},
        'between_splits_overlap': {}
    }

    # 检查每个split内部的重复数据
    for split in splits:
        duplicate_info = check_duplicates_within_split(split_data[split], split)
        results['within_split_duplicates'][split] = duplicate_info

        # 打印详细的重复信息
        if duplicate_info['total_duplicate_groups'] > 0:
            print(f"\n{split} 中的重复数据详情:")
            for (input_text, output_text), ids in duplicate_info['duplicate_details'].items():
                print(f"  重复数据组 (出现 {len(ids)} 次):")
                print(f"    Input: {input_text[:50]}..." if len(input_text) > 50 else f"    Input: {input_text}")
                print(f"    Output: {output_text[:50]}..." if len(output_text) > 50 else f"    Output: {output_text}")
                print(f"    IDs: {ids}")

    # 检查不同split之间的数据重合
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            split1, split2 = splits[i], splits[j]
            overlap_info = check_overlap_between_splits(
                split_data[split1], split_data[split2], split1, split2
            )
            results['between_splits_overlap'][f"{split1}_{split2}"] = overlap_info

            # 打印详细的重合信息
            if overlap_info['total_overlap'] > 0:
                print(f"\n{split1} 和 {split2} 之间的重合数据详情:")
                for idx, item in enumerate(overlap_info['overlapping_items'][:5]):  # 只显示前5个
                    print(f"  重合数据 {idx + 1}:")
                    print(f"    {split1} ID: {item['id_split1']}")
                    print(f"    {split2} ID: {item['id_split2']}")
                    print(f"    Input: {item['input'][:50]}..." if len(
                        item['input']) > 50 else f"    Input: {item['input']}")
                    print(f"    Output: {item['output'][:50]}..." if len(
                        item['output']) > 50 else f"    Output: {item['output']}")
                if overlap_info['total_overlap'] > 5:
                    print(f"    ... 还有 {overlap_info['total_overlap'] - 5} 条重合数据")

    return results


def print_summary(results: Dict):
    """打印分析结果摘要"""
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)

    print(f"数据目录: {results['data_dir']}")
    print(f"分析的splits: {results['splits_analyzed']}")

    print("\n各split内部重复情况:")
    for split, info in results['within_split_duplicates'].items():
        print(f"  {split}: {info['total_duplicate_groups']} 组重复数据, 共 {info['total_duplicate_items']} 条重复记录")

    print("\n不同split之间重合情况:")
    for split_pair, info in results['between_splits_overlap'].items():
        print(f"  {split_pair}: {info['total_overlap']} 条重合数据")


def main():
    parser = argparse.ArgumentParser(description='检查RetroSynthesis数据集的重合情况')
    parser.add_argument('--data_dir', type=str, default='/mnt/e/DataSets/Chemistry/RetroSynthesis',
                        help='数据目录路径')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='要检查的split列表')
    parser.add_argument('--data_file_pattern', type=str, default='retrosynthesis_{split}.json',
                        help='数据文件名模式，使用{split}作为占位符')

    args = parser.parse_args()

    try:
        results = analyze_dataset(
            data_dir=args.data_dir,
            splits=args.splits,
            data_file_pattern=args.data_file_pattern
        )

        print_summary(results)

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
