# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_extension.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/24 3:19

# https://chat.deepseek.com/a/chat/s/48a3b812-0062-4c48-9632-9a327296b12a

# !/usr/bin/env python3
"""
化学机制数据处理脚本
用于处理USPTO化学机制数据，生成扩展的JSON文件
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


class ChemicalMechanismProcessor:
    """化学机制数据处理器"""

    def __init__(self, data_dir: str = "/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/{split}",
                 data_file: str = "mech_USPTO.json",
                 output_file: str = "mech_USPTO_ext.json"):
        """
        初始化处理器

        Args:
            data_dir: 数据目录路径，包含{split}占位符
            data_file: 输入数据文件名
            output_file: 输出数据文件名
        """
        self.data_dir = data_dir
        self.data_file = data_file
        self.output_file = output_file

    def smiles_to_canonical(self, smiles: str, remove_atom_mapping: bool = False) -> str:
        """
        将SMILES字符串转换为规范形式

        Args:
            smiles: SMILES字符串
            remove_atom_mapping: 是否移除原子映射

        Returns:
            规范化的SMILES字符串
        """
        try:
            if remove_atom_mapping:
                # 移除原子映射：将类似[CH3:6]的模式转换为[CH3]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return smiles

                # 移除原子映射编号
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)

                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            else:
                # 保持原子映射
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return smiles
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            return canonical_smiles

        except Exception as e:
            print(f"Error processing SMILES: {smiles}, error: {e}")
            return smiles

    def sort_multiple_smiles(self, smiles: str) -> str:
        """
        对由"."分隔的多个SMILES进行排序（按原子编号）

        Args:
            smiles: 包含多个SMILES的字符串

        Returns:
            排序后的SMILES字符串
        """
        if "." not in smiles:
            return smiles

        # 分割多个SMILES
        parts = smiles.split(".")

        # 提取每个部分的第一个原子编号用于排序
        def get_first_atom_map(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol and mol.GetNumAtoms() > 0:
                    # 获取第一个原子的映射编号
                    first_atom = mol.GetAtomWithIdx(0)
                    return first_atom.GetAtomMapNum()
            except:
                pass
            return float('inf')  # 如果解析失败，放在最后

        # 按第一个原子的映射编号排序
        sorted_parts = sorted(parts, key=get_first_atom_map)

        return ".".join(sorted_parts)

    def process_reaction_string(self, reaction_smiles: str) -> Dict[str, str]:
        """
        处理反应SMILES字符串

        Args:
            reaction_smiles: 反应SMILES字符串（包含>>分隔符）

        Returns:
            包含各种处理结果的字典
        """
        try:
            # 分割反应物和产物
            if ">>" not in reaction_smiles:
                raise ValueError(f"Invalid reaction SMILES: {reaction_smiles}")

            reactants, products = reaction_smiles.split(">>")

            # 处理原子映射版本
            reactants_canonical = self.smiles_to_canonical(reactants, remove_atom_mapping=False)
            products_canonical = self.smiles_to_canonical(products, remove_atom_mapping=False)

            # 排序多个SMILES
            reactants_canonical = self.sort_multiple_smiles(reactants_canonical)
            products_canonical = self.sort_multiple_smiles(products_canonical)

            # 处理非原子映射版本
            reactants_no_map = self.smiles_to_canonical(reactants, remove_atom_mapping=True)
            products_no_map = self.smiles_to_canonical(products, remove_atom_mapping=True)

            # 排序多个SMILES（非原子映射版本）
            reactants_no_map = self.sort_multiple_smiles(reactants_no_map)
            products_no_map = self.sort_multiple_smiles(products_no_map)

            return {
                "amrxts": reactants,
                "amprds": products,
                "amrxts_cano": reactants_canonical,
                "amprds_cano": products_canonical,
                "rxts_cano": reactants_no_map,
                "prds_cano": products_no_map
            }

        except Exception as e:
            print(f"Error processing reaction: {reaction_smiles}, error: {e}")
            # 返回原始值作为fallback
            reactants, products = reaction_smiles.split(">>")
            return {
                "amrxts": reactants,
                "amprds": products,
                "amrxts_cano": reactants,
                "amprds_cano": products,
                "rxts_cano": reactants,
                "prds_cano": products
            }

    def process_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        处理单个数据文件

        Args:
            file_path: 输入文件路径

        Returns:
            处理后的数据列表
        """
        print(f"Processing file: {file_path}")

        # 读取原始数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} records from {file_path}")

        processed_data = []
        skipped_count = 0

        # 使用进度条处理每条记录
        for item in tqdm(data, desc="Processing records"):
            try:
                # 处理原始反应
                original_processed = self.process_reaction_string(item["original_reactions"])

                # 处理更新后的反应
                updated_processed = self.process_reaction_string(item["updated_reaction"])

                # 创建新的数据项
                new_item = {
                    "id": item["id"],
                    "original_reactions": item["original_reactions"],
                    "updated_reaction": item["updated_reaction"],

                    "mechanistic_class": item["mechanistic_class"],
                    "mechanistic_label": item["mechanistic_label"],

                    # 原始反应信息
                    "amrxts_in_ori": original_processed["amrxts"],
                    "amprds_in_ori": original_processed["amprds"],
                    "amrxts_cano_in_ori": original_processed["amrxts_cano"],
                    "amprds_cano_in_ori": original_processed["amprds_cano"],
                    "rxts_cano_in_ori": original_processed["rxts_cano"],
                    "prds_cano_in_ori": original_processed["prds_cano"],

                    # 更新反应信息
                    "amrxts_in_upd": updated_processed["amrxts"],
                    "amprds_in_upd": updated_processed["amprds"],
                    "amrxts_cano_in_upd": updated_processed["amrxts_cano"],
                    "amprds_cano_in_upd": updated_processed["amprds_cano"],
                    "rxts_cano_in_upd": updated_processed["rxts_cano"],
                    "prds_cano_in_upd": updated_processed["prds_cano"]
                }

                processed_data.append(new_item)

            except Exception as e:
                print(f"Error processing item {item.get('id', 'unknown')}: {e}")
                skipped_count += 1
                continue

        print(f"Successfully processed {len(processed_data)} records, skipped {skipped_count} records")

        return processed_data

    def process_all_splits(self, splits: List[str] = None):
        """
        处理所有数据分割

        Args:
            splits: 要处理的数据分割列表
        """
        if splits is None:
            splits = ["train", "val", "test"]

        for split in splits:
            print(f"\n{'=' * 60}")
            print(f"Processing {split} split")
            print(f"{'=' * 60}")

            # 构建文件路径
            input_dir = self.data_dir.format(split=split)
            input_file = os.path.join(input_dir, self.data_file)
            output_file = os.path.join(input_dir, self.output_file)

            # 检查输入文件是否存在
            if not os.path.exists(input_file):
                print(f"Warning: Input file {input_file} does not exist, skipping...")
                continue

            # 处理数据
            processed_data = self.process_single_file(input_file)

            # 保存处理后的数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            print(f"Saved processed data to: {output_file}")
            print(f"Total records processed for {split}: {len(processed_data)}")

            # 显示一些统计信息
            if processed_data:
                sample_item = processed_data[0]
                print(f"Sample processed item keys: {list(sample_item.keys())}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='化学机制数据处理脚本')

    # 添加命令行参数
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/{split}',
                        help='数据目录路径，包含{split}占位符')
    parser.add_argument('--data_file', type=str, default='mech_USPTO.json',
                        help='输入数据文件名')
    parser.add_argument('--output_file', type=str, default='mech_USPTO_ext.json',
                        help='输出数据文件名')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val', 'test'],
                        help='要处理的数据分割列表')

    args = parser.parse_args()

    # 创建处理器实例
    processor = ChemicalMechanismProcessor(
        data_dir=args.data_dir,
        data_file=args.data_file,
        output_file=args.output_file
    )

    # 处理所有数据分割
    processor.process_all_splits(args.splits)

    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()
