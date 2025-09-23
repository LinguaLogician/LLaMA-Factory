# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: dataset_check.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/22 21:09
# https://chat.deepseek.com/a/chat/s/dc9c510f-0ccb-40bd-8566-5b1567d0fee7
# !/usr/bin/env python3
"""
检查两个化学数据集之间的重复数据
"""

import os
import json
import argparse
from typing import List, Dict, Set, Tuple
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


def load_data_file(file_path: str) -> List[Dict]:
    """
    加载JSON数据文件

    Args:
        file_path: 文件路径

    Returns:
        数据列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path}: {e}")
        return []


def process_split(data_dir1: str, data_file1_pattern: str,
                  data_dir2: str, data_file2_pattern: str,
                  split: str, use_tqdm: bool = True) -> Dict:
    """
    处理单个split的数据

    Args:
        data_dir1: 第一个数据目录
        data_file1_pattern: 第一个数据文件模式
        data_dir2: 第二个数据目录
        data_file2_pattern: 第二个数据文件模式
        split: 数据集分割名称
        use_tqdm: 是否显示进度条

    Returns:
        处理结果字典
    """
    print(f"\n处理 {split} 分割...")

    # 构建文件路径
    file1_path = os.path.join(data_dir1, data_file1_pattern.format(split=split))
    file2_path = os.path.join(data_dir2, data_file2_pattern.format(split=split))

    print(f"加载文件1: {file1_path}")
    print(f"加载文件2: {file2_path}")

    # 加载数据
    data1 = load_data_file(file1_path)
    data2 = load_data_file(file2_path)

    if not data1 or not data2:
        print(f"警告: {split} 分割的一个或两个文件为空或加载失败")
        return {}

    print(f"文件1数据条数: {len(data1)}")
    print(f"文件2数据条数: {len(data2)}")

    # 预处理data2数据，创建查找字典
    print("预处理data2数据...")
    prds_cano_to_ids = {}
    rxts_cano_to_ids = {}

    iterable = tqdm(data2, desc="处理data2") if use_tqdm else data2
    for item in iterable:
        prds_cano = item.get('prds_cano', '')
        rxts_cano = item.get('rxts_cano', '')
        item_id = item.get('id', '')

        if prds_cano:
            if prds_cano not in prds_cano_to_ids:
                prds_cano_to_ids[prds_cano] = []
            prds_cano_to_ids[prds_cano].append(item_id)

        if rxts_cano:
            if rxts_cano not in rxts_cano_to_ids:
                rxts_cano_to_ids[rxts_cano] = []
            rxts_cano_to_ids[rxts_cano].append(item_id)

    # 查找重复数据
    print("查找重复数据...")
    matches_input = []  # data1.input 匹配 data2.prds_cano
    matches_output = []  # data1.output 匹配 data2.rxts_cano
    total_matches = set()  # 总的匹配项（去重）

    iterable = tqdm(data1, desc="处理data1") if use_tqdm else data1
    for item1 in iterable:
        input_smiles = item1.get('input', '')
        output_smiles = item1.get('output', '')
        item1_id = item1.get('id', '')

        # 规范化SMILES
        input_cano = canonicalize_smiles(input_smiles) if input_smiles else None
        output_cano = canonicalize_smiles(output_smiles) if output_smiles else None

        # 检查input匹配
        if input_cano and input_cano in prds_cano_to_ids:
            match_info = {
                'data1_id': item1_id,
                'data1_input': input_smiles,
                'data1_input_cano': input_cano,
                'matched_data2_ids': prds_cano_to_ids[input_cano],
                'match_type': 'input_to_prds'
            }
            matches_input.append(match_info)
            total_matches.add(item1_id)

        # 检查output匹配
        if output_cano and output_cano in rxts_cano_to_ids:
            match_info = {
                'data1_id': item1_id,
                'data1_output': output_smiles,
                'data1_output_cano': output_cano,
                'matched_data2_ids': rxts_cano_to_ids[output_cano],
                'match_type': 'output_to_rxts'
            }
            matches_output.append(match_info)
            total_matches.add(item1_id)

    # 统计结果
    result = {
        'split': split,
        'data1_count': len(data1),
        'data2_count': len(data2),
        'matches_input_count': len(matches_input),
        'matches_output_count': len(matches_output),
        'total_unique_matches': len(total_matches),
        'matches_input': matches_input,
        'matches_output': matches_output,
        'remaining_data1_count': len(data1) - len(total_matches)
    }

    print(f"{split} 分割结果:")
    print(f"  - 总匹配数 (input→prds): {len(matches_input)}")
    print(f"  - 总匹配数 (output→rxts): {len(matches_output)}")
    print(f"  - 唯一匹配数据条数: {len(total_matches)}")
    print(f"  - 剩余唯一数据条数: {len(data1) - len(total_matches)}")

    return result


def check_cross_split_duplicates(all_results: Dict[str, Dict], splits: List[str]) -> Dict:
    """
    检查不同split之间是否有重复数据

    Args:
        all_results: 所有split的处理结果
        splits: split列表

    Returns:
        跨split重复数据检查结果
    """
    print(f"\n检查跨split重复数据...")

    # 收集所有匹配的规范化SMILES
    all_matched_smiles = {}

    for split, result in all_results.items():
        split_matches = set()

        # 收集input匹配的SMILES
        for match in result['matches_input']:
            smiles = match['data1_input_cano']
            if smiles:
                split_matches.add(smiles)

        # 收集output匹配的SMILES
        for match in result['matches_output']:
            smiles = match['data1_output_cano']
            if smiles:
                split_matches.add(smiles)

        all_matched_smiles[split] = split_matches

    # 检查跨split重复
    cross_duplicates = {}
    for i, split1 in enumerate(splits):
        for split2 in splits[i + 1:]:
            common_smiles = all_matched_smiles[split1] & all_matched_smiles[split2]
            if common_smiles:
                cross_duplicates[f"{split1}_{split2}"] = {
                    'common_count': len(common_smiles),
                    'common_smiles': list(common_smiles)
                }
                print(f"  - {split1} 和 {split2} 有 {len(common_smiles)} 个重复数据")
            else:
                print(f"  - {split1} 和 {split2} 没有重复数据")

    return cross_duplicates


def main(data_dir1: str, data_file1_pattern: str,
         data_dir2: str, data_file2_pattern: str,
         splits: List[str], use_tqdm: bool = True):
    """
    主处理函数

    Args:
        data_dir1: 第一个数据目录
        data_file1_pattern: 第一个数据文件模式
        data_dir2: 第二个数据目录
        data_file2_pattern: 第二个数据文件模式
        splits: 要处理的split列表
        use_tqdm: 是否显示进度条
    """
    print("开始处理化学数据集重复数据检查...")
    print(f"数据目录1: {data_dir1}")
    print(f"数据文件1模式: {data_file1_pattern}")
    print(f"数据目录2: {data_dir2}")
    print(f"数据文件2模式: {data_file2_pattern}")
    print(f"处理的分割: {splits}")

    # 处理每个split
    all_results = {}
    for split in splits:
        result = process_split(data_dir1, data_file1_pattern, data_dir2,
                               data_file2_pattern, split, use_tqdm)
        if result:
            all_results[split] = result

    # 检查跨split重复
    cross_duplicates = check_cross_split_duplicates(all_results, splits)

    # 汇总结果
    print(f"\n汇总结果:")
    print("=" * 50)
    total_data1 = 0
    total_matches = 0

    for split, result in all_results.items():
        total_data1 += result['data1_count']
        total_matches += result['total_unique_matches']
        print(f"{split}:")
        print(f"  - 数据条数: {result['data1_count']}")
        print(f"  - 匹配条数: {result['total_unique_matches']}")
        print(f"  - 匹配比例: {result['total_unique_matches'] / result['data1_count'] * 100:.2f}%")

    print(f"\n总体统计:")
    print(f"  - 总数据条数: {total_data1}")
    print(f"  - 总匹配条数: {total_matches}")
    print(f"  - 总体匹配比例: {total_matches / total_data1 * 100:.2f}%")
    print(f"  - 跨split重复组数: {len(cross_duplicates)}")


if __name__ == "__main__":
    # 设置默认参数
    DEFAULT_DATA_DIR1 = "/mnt/e/DataSets/Chemistry/RetroSynthesis"
    DEFAULT_DATA_FILE1 = "retrosynthesis_{split}.json"
    DEFAULT_DATA_DIR2 = "/mnt/e/DataSets/Chemistry/USPTO50k_RAW/processed"
    DEFAULT_DATA_FILE2 = "uspto50k_{split}.json"
    DEFAULT_SPLITS = ["train", "val", "test"]

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="检查化学数据集之间的重复数据")
    parser.add_argument("--data_dir1", type=str, default=DEFAULT_DATA_DIR1,
                        help="第一个数据目录路径")
    parser.add_argument("--data_file1", type=str, default=DEFAULT_DATA_FILE1,
                        help="第一个数据文件模式，使用{split}作为占位符")
    parser.add_argument("--data_dir2", type=str, default=DEFAULT_DATA_DIR2,
                        help="第二个数据目录路径")
    parser.add_argument("--data_file2", type=str, default=DEFAULT_DATA_FILE2,
                        help="第二个数据文件模式，使用{split}作为占位符")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                        help="要处理的数据分割列表")
    parser.add_argument("--no_tqdm", action="store_true",
                        help="禁用进度条显示")

    args = parser.parse_args()

    # 执行主函数
    main(data_dir1=args.data_dir1,
         data_file1_pattern=args.data_file1,
         data_dir2=args.data_dir2,
         data_file2_pattern=args.data_file2,
         splits=args.splits,
         use_tqdm=not args.no_tqdm)