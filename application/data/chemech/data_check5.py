# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_check5.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/23 13:41

# !/usr/bin/env python3
"""
检查USPTO50k数据集中不同split之间的数据重合情况
支持三种不同的重合判断方式
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import sys


def load_data(file_path: Path) -> List[Dict]:
    """加载JSON数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path}: {e}")
        sys.exit(1)


def check_overlap_method1(data1: List[Dict], data2: List[Dict]) -> Set[str]:
    """方式1: 通过id判断重合"""
    ids1 = {item['id'] for item in data1 if 'id' in item}
    ids2 = {item['id'] for item in data2 if 'id' in item}
    return ids1.intersection(ids2)


def check_overlap_method2(data1: List[Dict], data2: List[Dict]) -> Set[Tuple]:
    """方式2: 通过amrxts_cano和amprds_cano判断重合"""
    keys1 = {(item['amrxts_cano'], item['amprds_cano'])
             for item in data1 if 'amrxts_cano' in item and 'amprds_cano' in item}
    keys2 = {(item['amrxts_cano'], item['amprds_cano'])
             for item in data2 if 'amrxts_cano' in item and 'amprds_cano' in item}
    return keys1.intersection(keys2)


def check_overlap_method3(data1: List[Dict], data2: List[Dict]) -> Set[Tuple]:
    """方式3: 通过rxts_cano和prds_cano判断重合"""
    keys1 = {(item['rxts_cano'], item['prds_cano'])
             for item in data1 if 'rxts_cano' in item and 'prds_cano' in item}
    keys2 = {(item['rxts_cano'], item['prds_cano'])
             for item in data2 if 'rxts_cano' in item and 'prds_cano' in item}
    return keys1.intersection(keys2)


def analyze_split_data(split_data: Dict[str, List[Dict]]) -> None:
    """分析每个split的数据基本情况"""
    print("\n" + "=" * 50)
    print("各Split数据统计")
    print("=" * 50)

    for split, data in split_data.items():
        valid_count = len([item for item in data if
                           all(key in item for key in ['id', 'amrxts_cano', 'amprds_cano', 'rxts_cano', 'prds_cano'])])
        print(f"{split}: 总数据量={len(data)}, 有效数据量={valid_count}")


def compare_splits(split_data: Dict[str, List[Dict]], split1: str, split2: str) -> None:
    """比较两个split之间的数据重合情况"""
    print(f"\n比较 {split1} 和 {split2}:")
    print("-" * 30)

    data1 = split_data[split1]
    data2 = split_data[split2]

    # 方式1: 通过id判断
    overlap_ids = check_overlap_method1(data1, data2)
    print(f"方式1 (ID相同): {len(overlap_ids)} 条重合数据")

    # 方式2: 通过amrxts_cano和amprds_cano判断
    overlap_method2 = check_overlap_method2(data1, data2)
    print(f"方式2 (amrxts_cano和amprds_cano相同): {len(overlap_method2)} 条重合数据")

    # 方式3: 通过rxts_cano和prds_cano判断
    overlap_method3 = check_overlap_method3(data1, data2)
    print(f"方式3 (rxts_cano和prds_cano相同): {len(overlap_method3)} 条重合数据")

    # 显示一些示例重合数据
    if overlap_ids:
        print(f"\n示例重合ID (前5个): {list(overlap_ids)[:5]}")


def main(data_dir: str, data_file_pattern: str, splits: List[str]) -> None:
    """主函数"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)

    # 加载所有split的数据
    print("正在加载数据文件...")
    split_data = {}

    for split in tqdm(splits, desc="加载split数据"):
        file_path = data_path / data_file_pattern.format(split=split)
        if not file_path.exists():
            print(f"警告: 文件不存在: {file_path}")
            continue

        data = load_data(file_path)
        split_data[split] = data
        print(f"{split} split: 加载了 {len(data)} 条数据")

    if len(split_data) < 2:
        print("错误: 需要至少两个split的数据进行比较")
        sys.exit(1)

    # 分析数据基本情况
    analyze_split_data(split_data)

    # 比较所有split组合
    print("\n" + "=" * 50)
    print("数据重合分析结果")
    print("=" * 50)

    split_list = list(split_data.keys())
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            compare_splits(split_data, split_list[i], split_list[j])


if __name__ == "__main__":
    # 设置默认参数
    default_data_dir = "/mnt/e/DataSets/Chemistry/USPTO50k/processed"
    default_data_file = "uspto50k_{split}.json"
    default_splits = ["train", "test", "val"]

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="检查USPTO50k数据集中不同split之间的数据重合情况")
    parser.add_argument("--data_dir", type=str, default=default_data_dir,
                        help=f"数据目录路径 (默认: {default_data_dir})")
    parser.add_argument("--data_file", type=str, default=default_data_file,
                        help=f"数据文件名模式 (默认: {default_data_file})")
    parser.add_argument("--splits", type=str, nargs="+", default=default_splits,
                        help=f"要比较的split名称 (默认: {default_splits})")

    args = parser.parse_args()

    print("USPTO50k数据集重合数据检查")
    print("=" * 50)
    print(f"数据目录: {args.data_dir}")
    print(f"数据文件模式: {args.data_file}")
    print(f"比较的splits: {args.splits}")
    print("=" * 50)

    # 执行主函数
    main(args.data_dir, args.data_file, args.splits)

    print("\n分析完成!")
