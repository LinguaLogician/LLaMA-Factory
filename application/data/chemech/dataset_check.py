# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: dataset_check.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/22 21:09

# !/usr/bin/env python3
# https://chat.deepseek.com/a/chat/s/7c948ffa-1b16-4a62-962b-c715c94d8e51
"""
检查两个化学数据集之间的重合数据
"""

import os
import json
import argparse
from typing import List, Dict, Set
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


def canonicalize_smiles(smiles: str) -> str:
    """
    将SMILES字符串转换为规范形式

    Args:
        smiles: SMILES字符串

    Returns:
        规范化的SMILES字符串，如果无效则返回None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def load_data_file(file_path: str, description: str) -> List[Dict]:
    """
    加载数据文件

    Args:
        file_path: 文件路径
        description: 文件描述（用于日志）

    Returns:
        数据列表
    """
    print(f"正在加载 {description}: {file_path}")
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"成功加载 {len(data)} 条数据")
    return data


def check_overlap(data_dir1: str, data_file1_pattern: str,
                  data_dir2: str, data_file2_pattern: str,
                  splits: List[str] = None) -> Dict:
    """
    检查两个数据集在不同split间的重合情况

    Args:
        data_dir1: 数据集1目录
        data_file1_pattern: 数据集1文件名模式（包含{split}）
        data_dir2: 数据集2目录
        data_file2_pattern: 数据集2文件名模式（包含{split}）
        splits: 要检查的split列表

    Returns:
        包含重合统计信息的字典
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    results = {}

    for split in splits:
        print(f"\n{'=' * 50}")
        print(f"处理 split: {split}")
        print(f"{'=' * 50}")

        # 构建文件路径
        file1_path = os.path.join(data_dir1, data_file1_pattern.format(split=split))
        file2_path = os.path.join(data_dir2, data_file2_pattern.format(split=split))

        # 加载数据
        data1 = load_data_file(file1_path, f"数据集1-{split}")
        data2 = load_data_file(file2_path, f"数据集2-{split}")

        if not data1 or not data2:
            print(f"跳过 {split}，因为至少一个文件为空或不存在")
            continue

        # 预处理数据集2：创建查找字典
        print("预处理数据集2...")
        data2_prds_lookup = {}  # prds_cano -> 数据项
        data2_rxts_lookup = {}  # rxts_cano -> 数据项

        for item in tqdm(data2, desc="处理数据集2"):
            prds_cano = item.get('prds_cano', '')
            rxts_cano = item.get('rxts_cano', '')

            if prds_cano:
                data2_prds_lookup[prds_cano] = item
            if rxts_cano:
                data2_rxts_lookup[rxts_cano] = item

        # 检查重合
        print("检查数据重合...")
        overlap_count = 0
        overlap_details = []

        for item1 in tqdm(data1, desc=f"检查数据集1-{split}"):
            input_smiles = item1.get('input', '')
            output_smiles = item1.get('output', '')

            # 规范化SMILES
            input_cano = canonicalize_smiles(input_smiles) if input_smiles else None
            output_cano = canonicalize_smiles(output_smiles) if output_smiles else None

            overlap_reason = None
            matched_item2 = None

            # 检查条件1: data_file1的input == data_file2的prds_cano
            if input_cano and input_cano in data2_prds_lookup:
                overlap_reason = "input == prds_cano"
                matched_item2 = data2_prds_lookup[input_cano]

            # 检查条件2: data_file1的output == data_file2的rxts_cano
            elif output_cano and output_cano in data2_rxts_lookup:
                overlap_reason = "output == rxts_cano"
                matched_item2 = data2_rxts_lookup[output_cano]

            if overlap_reason and matched_item2:
                overlap_count += 1
                overlap_details.append({
                    'data1_item': item1,
                    'data2_item': matched_item2,
                    'reason': overlap_reason
                })

        # 统计结果
        total_data1 = len(data1)
        total_data2 = len(data2)
        overlap_percentage = (overlap_count / total_data1 * 100) if total_data1 > 0 else 0

        results[split] = {
            'total_data1': total_data1,
            'total_data2': total_data2,
            'overlap_count': overlap_count,
            'overlap_percentage': overlap_percentage,
            'overlap_details': overlap_details
        }

        print(f"\n{split} 结果:")
        print(f"  数据集1条数: {total_data1}")
        print(f"  数据集2条数: {total_data2}")
        print(f"  重合条数: {overlap_count}")
        print(f"  重合比例: {overlap_percentage:.2f}%")
        print(f"  剩余唯一数据条数: {total_data1 - overlap_count}")

    return results


def check_cross_split_overlap(results: Dict, all_splits: List[str]) -> Dict:
    """
    检查不同split间的重合情况

    Args:
        results: 各split的重合结果
        all_splits: 所有split列表

    Returns:
        跨split重合统计
    """
    print(f"\n{'=' * 50}")
    print("检查跨split重合情况")
    print(f"{'=' * 50}")

    cross_overlap = {}

    # 收集所有规范化的SMILES
    all_canonical_smiles = {}

    for split, result in results.items():
        if 'overlap_details' not in result:
            continue

        for detail in result['overlap_details']:
            data1_item = detail['data1_item']
            input_smiles = data1_item.get('input', '')
            output_smiles = data1_item.get('output', '')

            input_cano = canonicalize_smiles(input_smiles) if input_smiles else None
            output_cano = canonicalize_smiles(output_smiles) if output_smiles else None

            if input_cano:
                if input_cano not in all_canonical_smiles:
                    all_canonical_smiles[input_cano] = set()
                all_canonical_smiles[input_cano].add(split)

            if output_cano:
                if output_cano not in all_canonical_smiles:
                    all_canonical_smiles[output_cano] = set()
                all_canonical_smiles[output_cano].add(split)

    # 统计跨split重合
    cross_split_count = 0
    for smiles, splits_set in all_canonical_smiles.items():
        if len(splits_set) > 1:
            cross_split_count += 1
            cross_overlap[smiles] = list(splits_set)

    print(f"跨split重合的SMILES数量: {cross_split_count}")

    return {
        'cross_split_smiles': cross_overlap,
        'cross_split_count': cross_split_count
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查两个化学数据集之间的重合数据')

    # 设置默认参数
    parser.add_argument('--data_dir1', default='/mnt/e/DataSets/Chemistry/RetroSynthesis',
                        help='数据集1目录路径')
    parser.add_argument('--data_file1', default='retrosynthesis_{split}.json',
                        help='数据集1文件名模式（包含{split}）')
    parser.add_argument('--data_dir2', default='/mnt/e/DataSets/Chemistry/USPTO50k_RAW/processed',
                        help='数据集2目录路径')
    parser.add_argument('--data_file2', default='uspto50k_{split}.json',
                        help='数据集2文件名模式（包含{split}）')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='要检查的split列表')

    args = parser.parse_args()

    print("开始检查数据集重合...")
    print(f"数据集1: {args.data_dir1}/{args.data_file1}")
    print(f"数据集2: {args.data_dir2}/{args.data_file2}")
    print(f"检查的splits: {args.splits}")

    # 检查重合
    results = check_overlap(
        data_dir1=args.data_dir1,
        data_file1_pattern=args.data_file1,
        data_dir2=args.data_dir2,
        data_file2_pattern=args.data_file2,
        splits=args.splits
    )

    # 检查跨split重合
    cross_results = check_cross_split_overlap(results, args.splits)

    # 输出最终总结
    print(f"\n{'=' * 60}")
    print("最终总结")
    print(f"{'=' * 60}")

    for split in args.splits:
        if split in results:
            result = results[split]
            print(f"{split}:")
            print(f"  数据集1: {result['total_data1']}条")
            print(f"  数据集2: {result['total_data2']}条")
            print(f"  重合: {result['overlap_count']}条 ({result['overlap_percentage']:.2f}%)")

    print(f"\n跨split重合: {cross_results['cross_split_count']}个唯一SMILES")

    # 保存详细结果（可选）
    output_file = "overlap_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'parameters': vars(args),
            'split_results': results,
            'cross_split_results': cross_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()