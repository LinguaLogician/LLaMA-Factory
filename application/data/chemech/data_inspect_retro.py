# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_inspect_retro.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/23 15:47
# https://chat.deepseek.com/a/chat/s/9d162c40-e931-40d2-abb3-55ec9a3a5e6b

# !/usr/bin/env python3
"""
检查RetroSynthesis数据集中不同split之间的数据重合情况
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import argparse
from tqdm import tqdm


class DataOverlapChecker:
    def __init__(self, data_dir: str = "/mnt/e/DataSets/Chemistry/RetroSynthesis",
                 data_file: str = "retrosynthesis_{split}.json"):
        self.data_dir = Path(data_dir)
        self.data_file_pattern = data_file
        self.splits = ["train", "test", "val"]

    def _tuple_key_to_str(self, key) -> str:
        """将tuple键转换为字符串格式"""
        if isinstance(key, tuple):
            return str(key)
        return str(key)

    def _str_to_tuple_key(self, key_str: str):
        """将字符串转换回tuple键（用于可能的后续处理）"""
        if key_str.startswith('(') and key_str.endswith(')'):
            try:
                # 简单的tuple字符串解析
                return eval(key_str)
            except:
                return key_str
        return key_str

    def load_data(self, split: str) -> List[Dict[str, Any]]:
        """加载指定split的数据文件"""
        file_path = self.data_dir / self.data_file_pattern.format(split=split)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        print(f"正在加载 {split} 数据...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"{split} 数据加载完成，共 {len(data)} 条记录")
        return data

    def check_intra_file_duplicates(self, data: List[Dict[str, Any]], split: str) -> Dict[str, Dict]:
        """检查单个文件内的重复数据"""
        print(f"\n正在检查 {split} 文件内的重复数据...")

        # 策略B1: input和output都相同
        stratB1_dict = defaultdict(list)
        # 策略B2: input相同
        stratB2_dict = defaultdict(list)
        # 策略B3: output相同
        stratB3_dict = defaultdict(list)

        for item in tqdm(data, desc=f"处理 {split} 数据"):
            input_key = item.get('input', '')
            output_key = item.get('output', '')
            item_id = item.get('id', '')

            # 策略B1
            stratB1_key = (input_key, output_key)
            stratB1_dict[stratB1_key].append(item_id)

            # 策略B2
            stratB2_dict[input_key].append(item_id)

            # 策略B3
            stratB3_dict[output_key].append(item_id)

        # 找出重复的数据，并将tuple键转换为字符串
        intra_duplicates = {
            'stratB1': {self._tuple_key_to_str(k): v for k, v in stratB1_dict.items() if len(v) > 1},
            'stratB2': {self._tuple_key_to_str(k): v for k, v in stratB2_dict.items() if len(v) > 1},
            'stratB3': {self._tuple_key_to_str(k): v for k, v in stratB3_dict.items() if len(v) > 1}
        }

        return intra_duplicates

    def check_inter_file_overlap(self, data1: List[Dict[str, Any]], data2: List[Dict[str, Any]],
                                 split1: str, split2: str) -> Dict[str, Dict]:
        """检查两个文件之间的数据重合"""
        print(f"\n正在检查 {split1} 和 {split2} 之间的数据重合...")

        # 构建第一个文件的索引
        stratA1_set1 = set()  # (input, output)
        stratA2_set1 = set()  # input
        stratA3_set1 = set()  # output

        stratA1_dict1 = defaultdict(list)  # (input, output) -> ids
        stratA2_dict1 = defaultdict(list)  # input -> ids
        stratA3_dict1 = defaultdict(list)  # output -> ids

        for item in tqdm(data1, desc=f"构建 {split1} 索引"):
            input_key = item.get('input', '')
            output_key = item.get('output', '')
            item_id = item.get('id', '')

            stratA1_key = (input_key, output_key)
            stratA1_set1.add(stratA1_key)
            stratA1_dict1[stratA1_key].append(item_id)

            stratA2_set1.add(input_key)
            stratA2_dict1[input_key].append(item_id)

            stratA3_set1.add(output_key)
            stratA3_dict1[output_key].append(item_id)

        # 检查第二个文件与第一个文件的重合
        inter_overlap = {
            'stratA1': defaultdict(list),  # input和output都相同
            'stratA2': defaultdict(list),  # input相同
            'stratA3': defaultdict(list)  # output相同
        }

        for item in tqdm(data2, desc=f"检查 {split2} 重合"):
            input_key = item.get('input', '')
            output_key = item.get('output', '')
            item_id = item.get('id', '')

            stratA1_key = (input_key, output_key)
            if stratA1_key in stratA1_set1:
                # 将tuple键转换为字符串
                str_key = self._tuple_key_to_str(stratA1_key)
                inter_overlap['stratA1'][str_key].extend(stratA1_dict1[stratA1_key] + [item_id])

            if input_key in stratA2_set1:
                inter_overlap['stratA2'][input_key].extend(stratA2_dict1[input_key] + [item_id])

            if output_key in stratA3_set1:
                inter_overlap['stratA3'][output_key].extend(stratA3_dict1[output_key] + [item_id])

        # 将defaultdict转换为普通dict
        return {
            'stratA1': dict(inter_overlap['stratA1']),
            'stratA2': dict(inter_overlap['stratA2']),
            'stratA3': dict(inter_overlap['stratA3'])
        }

    def generate_summary(self, intra_results: Dict, inter_results: Dict) -> Dict[str, Any]:
        """生成汇总信息"""
        summary = {
            'intra_file_duplicates': {},
            'inter_file_overlap': {},
            'statistics': {
                'intra': {},
                'inter': {}
            }
        }

        # 处理文件内重复数据汇总
        for split, strategies in intra_results.items():
            summary['intra_file_duplicates'][split] = {}
            summary['statistics']['intra'][split] = {}

            for strategy, duplicates in strategies.items():
                summary['intra_file_duplicates'][split][strategy] = duplicates
                summary['statistics']['intra'][split][strategy] = {
                    'duplicate_groups': len(duplicates),
                    'total_duplicate_ids': sum(len(ids) for ids in duplicates.values())
                }

        # 处理文件间重合数据汇总
        for pair, strategies in inter_results.items():
            summary['inter_file_overlap'][pair] = {}
            summary['statistics']['inter'][pair] = {}

            for strategy, overlaps in strategies.items():
                summary['inter_file_overlap'][pair][strategy] = overlaps
                summary['statistics']['inter'][pair][strategy] = {
                    'overlap_groups': len(overlaps),
                    'total_overlap_ids': sum(len(ids) for ids in overlaps.values())
                }

        return summary

    def run_analysis(self) -> Dict[str, Any]:
        """运行完整的分析流程"""
        print("开始数据重合分析...")

        # 加载所有split的数据
        all_data = {}
        for split in self.splits:
            try:
                all_data[split] = self.load_data(split)
            except FileNotFoundError as e:
                print(f"警告: {e}，跳过该split")
                continue

        if len(all_data) < 2:
            raise ValueError("至少需要两个split的数据文件才能进行分析")

        # 检查文件内重复数据
        intra_results = {}
        for split, data in all_data.items():
            intra_results[split] = self.check_intra_file_duplicates(data, split)

            # 打印文件内重复统计
            for strategy, duplicates in intra_results[split].items():
                print(f"{split} - {strategy}: 发现 {len(duplicates)} 组重复数据")
                if duplicates:
                    first_key = next(iter(duplicates.keys()))
                    print(f"  示例: {first_key} -> IDs: {duplicates[first_key][:3]}...")

        # 检查文件间数据重合
        inter_results = {}
        split_names = list(all_data.keys())

        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                split1, split2 = split_names[i], split_names[j]
                pair_name = f"{split1}_{split2}"

                inter_results[pair_name] = self.check_inter_file_overlap(
                    all_data[split1], all_data[split2], split1, split2
                )

                # 打印文件间重合统计
                for strategy, overlaps in inter_results[pair_name].items():
                    print(f"{pair_name} - {strategy}: 发现 {len(overlaps)} 组重合数据")
                    if overlaps:
                        first_key = next(iter(overlaps.keys()))
                        print(f"  示例: {first_key} -> IDs: {overlaps[first_key][:3]}...")

        # 生成汇总信息
        summary = self.generate_summary(intra_results, inter_results)

        # 保存汇总文件
        output_file = self.data_dir / "summary.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n分析完成！汇总文件已保存至: {output_file}")

        # 打印最终统计
        self.print_final_statistics(summary)

        return summary

    def print_final_statistics(self, summary: Dict[str, Any]):
        """打印最终统计信息"""
        print("\n" + "=" * 50)
        print("最终统计结果")
        print("=" * 50)

        # 文件内重复统计
        print("\n文件内重复数据统计:")
        for split, strategies in summary['statistics']['intra'].items():
            print(f"\n{split}:")
            for strategy, stats in strategies.items():
                print(f"  {strategy}: {stats['duplicate_groups']} 组重复, 涉及 {stats['total_duplicate_ids']} 个ID")

        # 文件间重合统计
        print("\n文件间数据重合统计:")
        for pair, strategies in summary['statistics']['inter'].items():
            print(f"\n{pair}:")
            for strategy, stats in strategies.items():
                print(f"  {strategy}: {stats['overlap_groups']} 组重合, 涉及 {stats['total_overlap_ids']} 个ID")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="检查RetroSynthesis数据集中不同split之间的数据重合情况")
    parser.add_argument("--data_dir", type=str, default="/mnt/e/DataSets/Chemistry/RetroSynthesis",
                        help="数据目录路径")
    parser.add_argument("--data_file", type=str, default="retrosynthesis_{split}.json",
                        help="数据文件名模式，使用{split}作为占位符")

    args = parser.parse_args()

    try:
        # 创建检查器并运行分析
        checker = DataOverlapChecker(args.data_dir, args.data_file)
        checker.run_analysis()
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())