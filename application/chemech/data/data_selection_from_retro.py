# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_selection_from_retro.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/23 16:46
# https://chat.deepseek.com/a/chat/s/b7ee0bdc-520c-430e-9da5-222fa6ca654e

import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import hashlib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChemicalDataProcessor:
    def __init__(self,
                 data_dir1: str = "/mnt/e/DataSets/Chemistry/ChemicalMechanism/",
                 data_dir2: str = "/mnt/e/DataSets/Chemistry/RetroSynthesis/",
                 output_dir1: str = "/mnt/e/DataSets/Chemistry/ProcessedMechAndRetro/MechV1",
                 output_dir2: str = "/mnt/e/DataSets/Chemistry/ProcessedMechAndRetro/RetroV1",
                 data_file: str = "mech-USPTO-31k.json",
                 splits: List[str] = None):

        if splits is None:
            splits = ["train", "val", "test"]

        self.data_dir1 = Path(data_dir1)
        self.data_dir2 = Path(data_dir2)
        self.output_dir1 = Path(output_dir1)
        self.output_dir2 = Path(output_dir2)
        self.data_file = data_file
        self.splits = splits

        # 创建输出目录
        self.output_dir1.mkdir(parents=True, exist_ok=True)
        self.output_dir2.mkdir(parents=True, exist_ok=True)

        # 存储数据
        self.mech_data = []
        self.retro_data = {split: [] for split in splits}

        # 存储匹配结果
        self.matching_results = {split: {} for split in splits}

    def load_data(self):
        """加载机制数据和逆合成数据"""
        logger.info("开始加载数据...")

        # 加载机制数据
        mech_file_path = self.data_dir1 / self.data_file
        if not mech_file_path.exists():
            raise FileNotFoundError(f"机制数据文件不存在: {mech_file_path}")

        with open(mech_file_path, 'r', encoding='utf-8') as f:
            self.mech_data = json.load(f)
        logger.info(f"加载机制数据: {len(self.mech_data)} 条记录")

        # 加载逆合成数据
        for split in self.splits:
            retro_file_path = self.data_dir2 / f"retrosynthesis_{split}.json"
            if not retro_file_path.exists():
                logger.warning(f"逆合成数据文件不存在: {retro_file_path}")
                continue

            with open(retro_file_path, 'r', encoding='utf-8') as f:
                self.retro_data[split] = json.load(f)
            logger.info(f"加载 {split} 数据: {len(self.retro_data[split])} 条记录")

    def smiles_to_canonical(self, smiles: str) -> str:
        """将SMILES转换为规范形式，并按字母顺序排序多组分"""
        if not smiles or smiles.strip() == "":
            return ""

        try:
            # 分割多组分
            parts = smiles.split('.')
            canonical_parts = []

            for part in parts:
                mol = Chem.MolFromSmiles(part.strip())
                if mol:
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    canonical_parts.append(canonical_smiles)

            # 按字母顺序排序并重新组合
            canonical_parts.sort()
            return '.'.join(canonical_parts)

        except Exception as e:
            logger.warning(f"SMILES转换失败: {smiles}, 错误: {e}")
            return ""

    def remove_atom_mapping(self, smiles: str) -> str:
        """移除SMILES中的原子映射"""
        if not smiles:
            return ""

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # 移除原子映射
                for atom in mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom.ClearProp('molAtomMapNumber')
                return Chem.MolToSmiles(mol, canonical=True)
            return ""
        except Exception as e:
            logger.warning(f"移除原子映射失败: {smiles}, 错误: {e}")
            return ""

    def process_mech_entry(self, entry: Dict) -> Dict:
        """处理单个机制数据条目"""
        processed = entry.copy()

        # 处理原始反应
        if 'original_reactions' in entry:
            parts = entry['original_reactions'].split('>>')
            if len(parts) == 2:
                processed['amrxts_in_ori'] = parts[0]
                processed['amprds_in_ori'] = parts[1]

                # 转换为规范形式（带原子映射）
                processed['amrxts_cano_in_ori'] = self.smiles_to_canonical(parts[0])
                processed['amprds_cano_in_ori'] = self.smiles_to_canonical(parts[1])

                # 转换为非原子映射形式
                rxts_no_map = self.remove_atom_mapping(parts[0])
                prds_no_map = self.remove_atom_mapping(parts[1])

                processed['rxts_cano_in_ori'] = self.smiles_to_canonical(rxts_no_map)
                processed['prds_cano_in_ori'] = self.smiles_to_canonical(prds_no_map)

        # 处理更新后的反应
        if 'updated_reaction' in entry:
            parts = entry['updated_reaction'].split('>>')
            if len(parts) == 2:
                processed['amrxts_in_upd'] = parts[0]
                processed['amprds_in_upd'] = parts[1]

                # 转换为规范形式（带原子映射）
                processed['amrxts_cano_in_upd'] = self.smiles_to_canonical(parts[0])
                processed['amprds_cano_in_upd'] = self.smiles_to_canonical(parts[1])

                # 转换为非原子映射形式
                rxts_no_map = self.remove_atom_mapping(parts[0])
                prds_no_map = self.remove_atom_mapping(parts[1])

                processed['rxts_cano_in_upd'] = self.smiles_to_canonical(rxts_no_map)
                processed['prds_cano_in_upd'] = self.smiles_to_canonical(prds_no_map)

        return processed

    def process_retro_entry(self, entry: Dict) -> Dict:
        """处理单个逆合成数据条目"""
        processed = entry.copy()

        # 转换输入和输出为规范形式
        if 'input' in entry:
            processed['prds_cano'] = self.smiles_to_canonical(entry['input'])

        if 'output' in entry:
            processed['rxts_cano'] = self.smiles_to_canonical(entry['output'])

        return processed

    def match_strategies(self, retro_entry: Dict, mech_entry: Dict) -> Optional[str]:
        """使用四种策略进行匹配"""
        retro_rxts = retro_entry.get('rxts_cano', '')
        retro_prds = retro_entry.get('prds_cano', '')

        # Strategy 1: 匹配原始反应的reactants和products
        mech_rxts_ori = mech_entry.get('rxts_cano_in_ori', '')
        mech_prds_ori = mech_entry.get('prds_cano_in_ori', '')

        if retro_rxts == mech_rxts_ori and retro_prds == mech_prds_ori:
            return "strat1"

        # Strategy 2: 匹配更新反应的reactants和products
        mech_rxts_upd = mech_entry.get('rxts_cano_in_upd', '')
        mech_prds_upd = mech_entry.get('prds_cano_in_upd', '')

        if retro_rxts == mech_rxts_upd and retro_prds == mech_prds_upd:
            return "strat2"

        # Strategy 3: 只匹配reactants
        if retro_rxts == mech_rxts_upd:
            return "strat3"

        # Strategy 4: 只匹配products
        if retro_prds == mech_prds_upd:
            return "strat4"

        return None

    def process_split(self, split: str):
        """处理单个split的数据"""
        logger.info(f"开始处理 {split} 数据...")

        # 处理逆合成数据
        processed_retro_data = []
        for entry in tqdm(self.retro_data[split], desc=f"处理 {split} 逆合成数据"):
            processed_retro_data.append(self.process_retro_entry(entry))

        # 处理机制数据
        processed_mech_data = []
        for entry in tqdm(self.mech_data, desc=f"处理 {split} 机制数据"):
            processed_mech_data.append(self.process_mech_entry(entry))

        # 初始化匹配结果
        strat_folders = ["strat1", "strat2", "strat3", "strat4"]

        # 为每个策略创建输出目录
        for strat in strat_folders:
            (self.output_dir1 / strat).mkdir(parents=True, exist_ok=True)
            (self.output_dir2 / strat).mkdir(parents=True, exist_ok=True)
            (self.output_dir1 / "tidy" / strat).mkdir(parents=True, exist_ok=True)

        # 初始化剩余数据
        remaining_retro_data = processed_retro_data.copy()
        remaining_mech_data = processed_mech_data.copy()

        # 存储匹配映射
        retro_to_mech_map = defaultdict(list)
        mech_to_retro_map = defaultdict(list)

        # 按策略进行匹配
        for strat in strat_folders:
            logger.info(f"应用策略 {strat}...")

            matched_retro_indices = set()
            matched_mech_indices = set()
            new_matched_pairs = []

            # 查找匹配对
            for i, retro_entry in tqdm(enumerate(remaining_retro_data), total=len(remaining_retro_data), desc=f"Strategy {strat}"):
                for j, mech_entry in enumerate(remaining_mech_data):
                    strategy = self.match_strategies(retro_entry, mech_entry)
                    if strategy == strat:
                        new_matched_pairs.append((i, j, retro_entry, mech_entry))
                        matched_retro_indices.add(i)
                        matched_mech_indices.add(j)

            # 更新映射关系
            for i, j, retro_entry, mech_entry in new_matched_pairs:
                retro_id = retro_entry['id']
                mech_id = mech_entry['id']
                retro_to_mech_map[retro_id].append(mech_id)
                mech_to_retro_map[mech_id].append(retro_id)

            # 保存匹配的机制数据
            matched_mech_entries = []
            for i, j, retro_entry, mech_entry in new_matched_pairs:
                matched_entry = mech_entry.copy()
                matched_entry['retro_id'] = [retro_entry['id']]
                matched_entry['strategy'] = strat
                matched_mech_entries.append(matched_entry)

            # 保存到文件
            if matched_mech_entries:
                output_file = self.output_dir1 / strat / f"mech-USPTO_{split}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(matched_mech_entries, f, indent=2, ensure_ascii=False)

                # 保存精简版
                tidy_entries = []
                for entry in matched_mech_entries:
                    tidy_entry = {
                        'id': entry['id'],
                        'retro_id': entry['retro_id'],
                        'strategy': entry['strategy'],
                        'rxts_cano_in_ori': entry.get('rxts_cano_in_ori', ''),
                        'prds_cano_in_ori': entry.get('prds_cano_in_ori', ''),
                        'rxts_cano_in_upd': entry.get('rxts_cano_in_upd', ''),
                        'prds_cano_in_upd': entry.get('prds_cano_in_upd', ''),
                        'mechanistic_class': entry.get('mechanistic_class', ''),
                        'mechanistic_label': entry.get('mechanistic_label', '')
                    }
                    tidy_entries.append(tidy_entry)

                tidy_file = self.output_dir1 / "tidy" / strat / f"mech-USPTO_{split}.json"
                with open(tidy_file, 'w', encoding='utf-8') as f:
                    json.dump(tidy_entries, f, indent=2, ensure_ascii=False)

            # 保存剩余的逆合成数据
            remaining_retro_data = [entry for i, entry in enumerate(remaining_retro_data)
                                    if i not in matched_retro_indices]

            rest_retro_file = self.output_dir2 / strat / f"retrosynthesis_{split}_rest.json"
            with open(rest_retro_file, 'w', encoding='utf-8') as f:
                json.dump(remaining_retro_data, f, indent=2, ensure_ascii=False)

            # 保存剩余的机制数据
            remaining_mech_data = [entry for i, entry in enumerate(remaining_mech_data)
                                   if i not in matched_mech_indices]

            rest_mech_file = self.output_dir1 / strat / f"mech-USPTO_rest.json"
            with open(rest_mech_file, 'w', encoding='utf-8') as f:
                json.dump(remaining_mech_data, f, indent=2, ensure_ascii=False)

            # 保存精简版剩余机制数据
            tidy_rest_entries = []
            for entry in remaining_mech_data:
                tidy_entry = {
                    'id': entry['id'],
                    'rxts_cano_in_ori': entry.get('rxts_cano_in_ori', ''),
                    'prds_cano_in_ori': entry.get('prds_cano_in_ori', ''),
                    'rxts_cano_in_upd': entry.get('rxts_cano_in_upd', ''),
                    'prds_cano_in_upd': entry.get('prds_cano_in_upd', ''),
                    'mechanistic_class': entry.get('mechanistic_class', ''),
                    'mechanistic_label': entry.get('mechanistic_label', '')
                }
                tidy_rest_entries.append(tidy_entry)

            tidy_rest_file = self.output_dir1 / "tidy" / strat / "mech-USPTO_rest.json"
            with open(tidy_rest_file, 'w', encoding='utf-8') as f:
                json.dump(tidy_rest_entries, f, indent=2, ensure_ascii=False)

            logger.info(f"策略 {strat}: 匹配了 {len(matched_mech_entries)} 对数据")
            logger.info(f"策略 {strat}: 剩余逆合成数据 {len(remaining_retro_data)} 条")
            logger.info(f"策略 {strat}: 剩余机制数据 {len(remaining_mech_data)} 条")

        # 为逆合成数据添加匹配的机制ID并保存
        for strat in strat_folders:
            strat_retro_data = []
            for entry in processed_retro_data:
                retro_id = entry['id']
                if retro_id in retro_to_mech_map:
                    new_entry = entry.copy()
                    # 在id字段后插入mech_id字段
                    keys = list(new_entry.keys())
                    id_index = keys.index('id')
                    new_entry = {}
                    for i, key in enumerate(keys):
                        if i == id_index + 1:
                            new_entry['mech_id'] = retro_to_mech_map[retro_id]
                        new_entry[key] = entry[key]
                    strat_retro_data.append(new_entry)

            if strat_retro_data:
                output_file = self.output_dir2 / strat / f"retrosynthesis_{split}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(strat_retro_data, f, indent=2, ensure_ascii=False)

        # 保存摘要信息
        self.save_summary(split, retro_to_mech_map, mech_to_retro_map,
                          len(processed_retro_data), len(processed_mech_data))

        # 保存映射关系
        self.save_mappings(split, retro_to_mech_map, mech_to_retro_map)

    def save_summary(self, split: str, retro_to_mech: Dict, mech_to_retro: Dict,
                     total_retro: int, total_mech: int):
        """保存摘要信息"""
        summary = {
            "split": split,
            "total_retro_entries": total_retro,
            "total_mech_entries": total_mech,
            "matched_retro_entries": len(retro_to_mech),
            "matched_mech_entries": len(mech_to_retro),
            "unmatched_retro_entries": total_retro - len(retro_to_mech),
            "unmatched_mech_entries": total_mech - len(mech_to_retro),
            "matching_statistics": {
                "retro_entries_with_single_match": sum(1 for v in retro_to_mech.values() if len(v) == 1),
                "retro_entries_with_multiple_matches": sum(1 for v in retro_to_mech.values() if len(v) > 1),
                "mech_entries_with_single_match": sum(1 for v in mech_to_retro.values() if len(v) == 1),
                "mech_entries_with_multiple_matches": sum(1 for v in mech_to_retro.values() if len(v) > 1)
            }
        }

        # 保存到两个输出目录
        summary_file1 = self.output_dir1 / f"summary_{split}.json"
        summary_file2 = self.output_dir2 / f"summary_{split}.json"

        with open(summary_file1, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(summary_file2, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def save_mappings(self, split: str, retro_to_mech: Dict, mech_to_retro: Dict):
        """保存映射关系"""
        # 保存retro_id到mech_id的映射
        retro_mech_mapping = {
            "split": split,
            "mapping_type": "retro_id_to_mech_id",
            "mappings": retro_to_mech
        }

        mapping_file1 = self.output_dir1 / f"mapping_retro_to_mech_{split}.json"
        with open(mapping_file1, 'w', encoding='utf-8') as f:
            json.dump(retro_mech_mapping, f, indent=2, ensure_ascii=False)

        # 保存mech_id到retro_id的映射
        mech_retro_mapping = {
            "split": split,
            "mapping_type": "mech_id_to_retro_id",
            "mappings": mech_to_retro
        }

        mapping_file2 = self.output_dir2 / f"mapping_mech_to_retro_{split}.json"
        with open(mapping_file2, 'w', encoding='utf-8') as f:
            json.dump(mech_retro_mapping, f, indent=2, ensure_ascii=False)

    def process_all(self):
        """处理所有数据"""
        logger.info("开始处理所有数据...")

        # 加载数据
        self.load_data()

        # 处理每个split
        for split in self.splits:
            if split in self.retro_data and self.retro_data[split]:
                self.process_split(split)
            else:
                logger.warning(f"跳过 {split} 数据，无有效数据")

        logger.info("数据处理完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="化学机制与逆合成数据处理器")

    parser.add_argument('--data_dir1', type=str,
                        default="/mnt/e/DataSets/Chemistry/ChemicalMechanism/",
                        help='机制数据目录')
    parser.add_argument('--data_dir2', type=str,
                        default="/mnt/e/DataSets/Chemistry/RetroSynthesis/",
                        help='逆合成数据目录')
    parser.add_argument('--output_dir1', type=str,
                        default="/mnt/e/DataSets/Chemistry/ProcessedMechAndRetro/MechV1",
                        help='机制数据输出目录')
    parser.add_argument('--output_dir2', type=str,
                        default="/mnt/e/DataSets/Chemistry/ProcessedMechAndRetro/RetroV1",
                        help='逆合成数据输出目录')
    parser.add_argument('--data_file', type=str,
                        default="mech-USPTO-31k.json",
                        help='机制数据文件名')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=["train", "val", "test"],
                        help='要处理的数据split')

    args = parser.parse_args()

    # 创建处理器并执行
    processor = ChemicalDataProcessor(
        data_dir1=args.data_dir1,
        data_dir2=args.data_dir2,
        output_dir1=args.output_dir1,
        output_dir2=args.output_dir2,
        data_file=args.data_file,
        splits=args.splits
    )

    processor.process_all()


if __name__ == "__main__":
    main()