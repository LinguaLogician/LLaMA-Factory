# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_conversion.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/22 11:26
# https://chat.deepseek.com/a/chat/s/e41a37a5-d4b0-4a46-be4d-62d578132979

import os
import csv
import json
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


def canonicalize_smiles(smiles, remove_atom_mapping=True):
    """将SMILES字符串转换为canonical形式"""
    try:
        if remove_atom_mapping:
            # 移除原子映射并转换为canonical形式
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # 移除原子映射
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.SetAtomMapNum(0)
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            # 保持原子映射，只进行canonical化
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        print(f"Error canonicalizing SMILES: {smiles}, Error: {e}")
        return None


def process_reaction_smiles(reaction_smiles):
    """处理反应SMILES字符串"""
    try:
        # 拆分反应物和生成物
        if '>>' not in reaction_smiles:
            return None, None, None, None

        reactants, products = reaction_smiles.split('>>', 1)

        # 处理反应物（可能包含试剂，用'.'分隔）
        reactant_list = reactants.split('.')
        product_list = products.split('.')

        # 转换为canonical形式（保持原子映射）
        amrxts_cano_list = []
        for rxt in reactant_list:
            if rxt.strip():
                cano = canonicalize_smiles(rxt, remove_atom_mapping=False)
                if cano:
                    amrxts_cano_list.append(cano)

        amprds_cano_list = []
        for prd in product_list:
            if prd.strip():
                cano = canonicalize_smiles(prd, remove_atom_mapping=False)
                if cano:
                    amprds_cano_list.append(cano)

        # 排序并重新组合
        amrxts_cano = '.'.join(sorted(amrxts_cano_list))
        amprds_cano = '.'.join(sorted(amprds_cano_list))

        # 转换为非原子映射形式并canonical化
        rxts_cano_list = []
        for rxt in reactant_list:
            if rxt.strip():
                cano = canonicalize_smiles(rxt, remove_atom_mapping=True)
                if cano:
                    rxts_cano_list.append(cano)

        prds_cano_list = []
        for prd in product_list:
            if prd.strip():
                cano = canonicalize_smiles(prd, remove_atom_mapping=True)
                if cano:
                    prds_cano_list.append(cano)

        # 排序并重新组合
        rxts_cano = '.'.join(sorted(rxts_cano_list))
        prds_cano = '.'.join(sorted(prds_cano_list))

        return reactants, products, amrxts_cano, amprds_cano, rxts_cano, prds_cano

    except Exception as e:
        print(f"Error processing reaction: {reaction_smiles}, Error: {e}")
        return None, None, None, None, None, None


def process_csv_to_json(data_dir, output_dir_name, splits=['train', 'val', 'test']):
    """将CSV文件转换为JSON格式"""

    # 创建输出目录
    output_dir = os.path.join(data_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for split in splits:
        input_file = os.path.join(data_dir, f'raw_{split}.csv')
        output_file = os.path.join(output_dir, f'raw_{split}.json')

        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            continue

        print(f"Processing {split} data...")

        json_data = []
        skipped_count = 0

        # 读取CSV文件
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            total_rows = sum(1 for _ in reader)
            csvfile.seek(0)  # 重置文件指针
            next(reader)  # 跳过表头

            # 使用tqdm显示进度条
            for row in tqdm(reader, total=total_rows - 1, desc=f"Processing {split}"):
                try:
                    reaction_smiles = row['reactants>reagents>production']

                    # 处理反应SMILES
                    amrxts, amprds, amrxts_cano, amprds_cano, rxts_cano, prds_cano = process_reaction_smiles(
                        reaction_smiles)

                    if amrxts is None:
                        skipped_count += 1
                        continue

                    # 创建新的数据条目
                    entry = {
                        'id': row['id'],
                        'class': row['class'],
                        'reaction': reaction_smiles,
                        'amrxts': amrxts,
                        'amprds': amprds,
                        'amrxts_cano': amrxts_cano,
                        'amprds_cano': amprds_cano,
                        'rxts_cano': rxts_cano,
                        'prds_cano': prds_cano
                    }

                    json_data.append(entry)

                except Exception as e:
                    print(f"Error processing row {row.get('id', 'unknown')}: {e}")
                    skipped_count += 1
                    continue

        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)

        print(f"Processed {len(json_data)} entries for {split}, skipped {skipped_count} entries")
        print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Process USPTO-50k dataset from CSV to JSON format')
    parser.add_argument('--data_dir', type=str, default='/mnt/e/DataSets/Chemistry/USPTO50k_RAW',
                        help='Directory containing the CSV files')
    parser.add_argument('--output_dir', type=str, default='processed',
                        help='Output directory name (default: processed)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Dataset splits to process (default: train val test)')

    args = parser.parse_args()

    print(f"Starting processing...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits: {args.splits}")

    process_csv_to_json(args.data_dir, args.output_dir, args.splits)

    print("Processing completed!")


if __name__ == '__main__':
    main()