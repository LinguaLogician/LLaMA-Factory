# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_selection.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/20 20:27
# https://chat.deepseek.com/a/chat/s/af77ad5d-c45c-4745-84bc-0991f8a3c6df

import json
import os
from argparse import ArgumentParser
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


def canonicalize_smiles(smiles: str) -> str:
    """
    将SMILES字符串转为canonical形式。
    如果包含多个分子（用'.'分隔），则分别转换后按字母顺序排序再拼接。
    """
    if not smiles:
        return ""
    parts = smiles.split('.')
    canonical_parts = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            canonical_parts.append(part)
        else:
            canonical_parts.append(Chem.MolToSmiles(mol, canonical=True))
    canonical_parts.sort()
    return '.'.join(canonical_parts)


def remove_atom_mapping(smiles: str) -> str:
    """
    移除SMILES字符串中的原子映射（如[CH3:1] -> [CH3]）。
    注意：保留原子类型和括号。
    """
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol, canonical=False)


def process_mechanism_data(data_file_path: str) -> List[Dict[str, Any]]:
    """
    处理data_file（mech-USPTO-31k.json）中的数据，生成所需字段。
    """
    with open(data_file_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for item in tqdm(data, desc="Processing mechanism data"):
        # 拆分original_reactions
        original_rxn = item['original_reactions']
        if '>>' in original_rxn:
            amrxts_in_ori, amprds_in_ori = original_rxn.split('>>')
        else:
            amrxts_in_ori, amprds_in_ori = original_rxn, ""

        # 拆分updated_reaction
        updated_rxn = item['updated_reaction']
        if '>>' in updated_rxn:
            amrxts_in_upd, amprds_in_upd = updated_rxn.split('>>')
        else:
            amrxts_in_upd, amprds_in_upd = updated_rxn, ""

        # 转换为canonical形式（保留原子映射）
        amrxts_cano_in_ori = canonicalize_smiles(amrxts_in_ori)
        amprds_cano_in_ori = canonicalize_smiles(amprds_in_ori)
        amrxts_cano_in_upd = canonicalize_smiles(amrxts_in_upd)
        amprds_cano_in_upd = canonicalize_smiles(amprds_in_upd)

        # 移除原子映射并转换为canonical形式
        rxts_noam_ori = remove_atom_mapping(amrxts_in_ori)
        prds_noam_ori = remove_atom_mapping(amprds_in_ori)
        rxts_noam_upd = remove_atom_mapping(amrxts_in_upd)
        prds_noam_upd = remove_atom_mapping(amprds_in_upd)

        rxts_cano_in_ori = canonicalize_smiles(rxts_noam_ori)
        prds_cano_in_ori = canonicalize_smiles(prds_noam_ori)
        rxts_cano_in_upd = canonicalize_smiles(rxts_noam_upd)
        prds_cano_in_upd = canonicalize_smiles(prds_noam_upd)

        processed_item = {
            **item,
            'amrxts_in_ori': amrxts_in_ori,
            'amprds_in_ori': amprds_in_ori,
            'amrxts_in_upd': amrxts_in_upd,
            'amprds_in_upd': amprds_in_upd,
            'amrxts_cano_in_ori': amrxts_cano_in_ori,
            'amprds_cano_in_ori': amprds_cano_in_ori,
            'amrxts_cano_in_upd': amrxts_cano_in_upd,
            'amprds_cano_in_upd': amprds_cano_in_upd,
            'rxts_cano_in_ori': rxts_cano_in_ori,
            'prds_cano_in_ori': prds_cano_in_ori,
            'rxts_cano_in_upd': rxts_cano_in_upd,
            'prds_cano_in_upd': prds_cano_in_upd
        }
        processed_data.append(processed_item)

    return processed_data


def match_strategy(uspto_item: Dict[str, Any], mech_item: Dict[str, Any], strategy: int) -> bool:
    """
    根据策略判断两个条目是否匹配。
    """
    if strategy == 1:
        # strat1: reaction (USPTO) vs original_reactions (mech)
        return uspto_item['reaction'] == mech_item['original_reactions']
    elif strategy == 2:
        # strat2: reaction (USPTO) vs updated_reaction (mech)
        return uspto_item['reaction'] == mech_item['updated_reaction']
    elif strategy == 3:
        # strat3: amrxts & amprds (USPTO) vs amrxts_in_ori & amprds_in_ori (mech)
        return (uspto_item['amrxts'] == mech_item['amrxts_in_ori'] and
                uspto_item['amprds'] == mech_item['amprds_in_ori'])
    elif strategy == 4:
        # strat4: amrxts & amprds (USPTO) vs amrxts_in_upd & amprds_in_upd (mech)
        return (uspto_item['amrxts'] == mech_item['amrxts_in_upd'] and
                uspto_item['amprds'] == mech_item['amprds_in_upd'])
    elif strategy == 5:
        # strat5: amrxts_cano & amprds_cano (USPTO) vs amrxts_cano_in_ori & amprds_cano_in_ori (mech)
        return (uspto_item['amrxts_cano'] == mech_item['amrxts_cano_in_ori'] and
                uspto_item['amprds_cano'] == mech_item['amprds_cano_in_ori'])
    elif strategy == 6:
        # strat6: amrxts_cano & amprds_cano (USPTO) vs amrxts_cano_in_upd & amprds_cano_in_upd (mech)
        return (uspto_item['amrxts_cano'] == mech_item['amrxts_cano_in_upd'] and
                uspto_item['amprds_cano'] == mech_item['amprds_cano_in_upd'])
    elif strategy == 7:
        # strat7: rxts_cano & prds_cano (USPTO) vs rxts_cano_in_ori & prds_cano_in_ori (mech)
        return (uspto_item['rxts_cano'] == mech_item['rxts_cano_in_ori'] and
                uspto_item['prds_cano'] == mech_item['prds_cano_in_ori'])
    elif strategy == 8:
        # strat8: rxts_cano & prds_cano (USPTO) vs rxts_cano_in_upd & prds_cano_in_upd (mech)
        return (uspto_item['rxts_cano'] == mech_item['rxts_cano_in_upd'] and
                uspto_item['prds_cano'] == mech_item['prds_cano_in_upd'])
    else:
        return False


def main(data_dir1: str, data_dir2: str, data_file: str, split_files: List[str]):
    # 读取并处理机制数据
    mech_data_path = os.path.join(data_dir1, data_file)
    processed_mech_data = process_mechanism_data(mech_data_path)
    total_mech = len(processed_mech_data)

    # 初始化未匹配的机制数据集合（用索引表示）
    unmatched_mech_indices = set(range(total_mech))

    for split in split_files:
        print(f"\nProcessing split: {split}")
        # 读取USPTO split数据
        uspto_file_path = os.path.join(data_dir2, f"uspto50k_{split}.json")
        with open(uspto_file_path, 'r') as f:
            uspto_data = json.load(f)

        total_uspto = len(uspto_data)
        unmatched_uspto_indices = set(range(total_uspto))

        # 为每个策略创建输出目录和列表
        output_dirs = {}
        output_data = {}
        for strat in range(1, 9):
            output_dir = os.path.join(data_dir1, args.version, f"strat{strat}")
            os.makedirs(output_dir, exist_ok=True)
            output_dirs[strat] = output_dir
            output_data[strat] = []

        # 遍历每个策略（按优先级从高到低）
        for strat in range(1, 9):
            matched_uspto_indices = set()
            matched_mech_indices = set()

            # 遍历未匹配的USPTO数据
            for uspto_idx in tqdm(unmatched_uspto_indices, desc=f"Strategy {strat}"):
                uspto_item = uspto_data[uspto_idx]
                found = False

                # 在未匹配的机制数据中查找
                for mech_idx in unmatched_mech_indices:
                    mech_item = processed_mech_data[mech_idx]
                    if match_strategy(uspto_item, mech_item, strat):
                        # 匹配成功
                        matched_uspto_indices.add(uspto_idx)
                        matched_mech_indices.add(mech_idx)

                        # 创建新的条目
                        new_item = {
                            **mech_item,
                            'id': mech_item['id'],
                            'uspto50k_id': uspto_item['id'],
                            'strategy': f'strat{strat}'
                        }
                        output_data[strat].append(new_item)
                        found = True
                        break

            # 更新未匹配的索引
            unmatched_uspto_indices -= matched_uspto_indices
            unmatched_mech_indices -= matched_mech_indices

            # 保存当前策略的匹配结果
            output_file = os.path.join(output_dirs[strat], f"mech-USPTO_{split}.json")
            with open(output_file, 'w') as f:
                json.dump(output_data[strat], f, indent=2)

            print(
                f"Strategy {strat}: matched {len(matched_uspto_indices)} USPTO items, {len(matched_mech_indices)} mech items")

        # 保存未匹配的USPTO数据
        rest_uspto = [uspto_data[i] for i in unmatched_uspto_indices]
        rest_uspto_file = os.path.join(data_dir1, args.version, f"uspto50k_{split}_rest.json")
        with open(rest_uspto_file, 'w') as f:
            json.dump(rest_uspto, f, indent=2)

        print(f"Split {split}: {len(rest_uspto)} USPTO items unmatched")

    # 保存未匹配的机制数据
    rest_mech = [processed_mech_data[i] for i in unmatched_mech_indices]
    rest_mech_file = os.path.join(data_dir1, args.version, "mech-USPTO_rest.json")
    with open(rest_mech_file, 'w') as f:
        json.dump(rest_mech, f, indent=2)

    print(f"\nTotal unmatched mech items: {len(rest_mech)}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Match USPTO50k data with mechanism data")
    parser.add_argument("--data_dir1", default="/mnt/e/DataSets/Chemistry/Chemech",
                        help="Directory containing mechanism data")
    parser.add_argument("--data_dir2", default="/mnt/e/DataSets/Chemistry/USPTO50k/processed",
                        help="Directory containing USPTO50k split data")
    parser.add_argument("--data_file", default="mech-USPTO-31k.json", help="Mechanism data file name")
    parser.add_argument("--version", default="v1", help="Version")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to process")

    args = parser.parse_args()

    main(args.data_dir1, args.data_dir2, args.data_file, args.splits)