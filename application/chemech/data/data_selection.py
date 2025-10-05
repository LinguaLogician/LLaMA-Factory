# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_selection.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/20 20:27
# https://chat.deepseek.com/a/chat/s/2567f80d-1de4-4a49-a99f-26c97ab7e981
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


def canonical_simple_smiles(smiles):
    """将SMILES转换为规范的简单SMILES格式（无原子映射）"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 移除原子映射
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def process_reaction_smiles(reaction_smiles):
    """处理反应SMILES，分割反应物和产物并转换为规范SMILES"""
    if '>>' not in reaction_smiles:
        return None, None

    reactants, products = reaction_smiles.split('>>')

    # 处理多个反应物（用'.'分隔）
    reactant_smiles_list = []
    for r_smiles in reactants.split('.'):
        canonical_r = canonical_simple_smiles(r_smiles)
        if canonical_r:
            reactant_smiles_list.append(canonical_r)

    product_smiles_list = []
    for p_smiles in products.split('.'):
        canonical_p = canonical_simple_smiles(p_smiles)
        if canonical_p:
            product_smiles_list.append(canonical_p)

    # 排序并连接以确保一致性
    reactant_smiles_list.sort()
    product_smiles_list.sort()

    return '.'.join(reactant_smiles_list), '.'.join(product_smiles_list)


def process_smiles_string(smiles_string):
    # 使用正则表达式提取所有英文字母
    letters = re.findall(r'[a-zA-Z]', smiles_string)

    # 将所有字母转换为小写
    lowercase_letters = [letter.lower() for letter in letters]

    # 对字母进行排序
    sorted_letters = sorted(lowercase_letters)

    return sorted_letters

def main(data_dir1, data_dir2, data_file, split_names=None):
    if split_names is None:
        split_names = ['train', 'valid', 'test']

    data_dir1 = Path(data_dir1)
    data_dir2 = Path(data_dir2)

    # 加载机制数据
    print("Loading mechanism data...")
    with open(data_dir1 / data_file, 'r', encoding='utf-8') as f:
        mechanism_data = json.load(f)

    print(f"Loaded {len(mechanism_data)} mechanism records")

    # 预处理机制数据
    print("Preprocessing mechanism data...")
    mechanism_preprocessed = []
    for item in tqdm(mechanism_data, desc="Processing mechanism data"):
        try:
            ori_reactants, ori_products = process_reaction_smiles(item['original_reactions'])
            reactants, products = process_reaction_smiles(item['updated_reaction'])
            if reactants and products:
                mechanism_preprocessed.append({
                    'original_id': item['id'],
                    'reactants': reactants,
                    'products': products,
                    'ori_reactants': ori_reactants,
                    'ori_products': ori_products,
                    'original_data': item
                })
        except:
            continue

    print(f"Successfully preprocessed {len(mechanism_preprocessed)} mechanism records")

    # 处理每个split
    for split in split_names:
        print(f"\nProcessing {split} split...")

        # 加载逆合成数据
        retro_file = data_dir2 / f"retrosynthesis_{split}.json"
        if not retro_file.exists():
            print(f"Warning: {retro_file} does not exist, skipping")
            continue

        with open(retro_file, 'r', encoding='utf-8') as f:
            retro_data = json.load(f)

        print(f"Loaded {len(retro_data)} {split} records")

        # 预处理逆合成数据
        retro_preprocessed = []
        for item in tqdm(retro_data, desc=f"Processing {split} data"):
            try:
                input_smiles = canonical_simple_smiles(item['input'])
                output_smiles = canonical_simple_smiles(item['output'])
                if input_smiles and output_smiles:
                    retro_preprocessed.append({
                        'original_id': item['id'],
                        'input': input_smiles,
                        'output': output_smiles,
                        'original_data': item
                    })
            except:
                continue

        print(f"Successfully preprocessed {len(retro_preprocessed)} {split} records")

        # 匹配数据
        matched_mechanism = []
        matched_retro = set()
        rest_mechanism = mechanism_preprocessed.copy()

        for retro_item in tqdm(retro_preprocessed, desc=f"Matching {split} data"):
            found_match = False
            for i, mech_item in enumerate(rest_mechanism):
                if (
                        (
                        (retro_item['input']) == (mech_item['products']) and
                        (retro_item['output']) == (mech_item['reactants']))
                        or
                        ((retro_item['output']) == (mech_item['ori_reactants']) and
                        (retro_item['input']) == (mech_item['ori_products']))
                ):
                    # 创建匹配的数据项
                    matched_data = mech_item['original_data'].copy()
                    matched_data['id'] = matched_data['id'].replace('chemical_mechanism', f'chemical_mechanism_{split}')
                    matched_data['retrosyn_id'] = retro_item['original_id']
                    matched_mechanism.append(matched_data)

                    matched_retro.add(retro_item['original_id'])
                    rest_mechanism.pop(i)
                    found_match = True
                    break

            # 如果没有找到匹配，继续检查下一个

        # 保存匹配的机制数据
        output_file = data_dir1 / f"mech-USPTO-31k_{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matched_mechanism, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(matched_mechanism)} matched records to {output_file}")

        # 保存未匹配的逆合成数据
        rest_retro = [item for item in retro_preprocessed
                      if item['original_id'] not in matched_retro]

        rest_retro_file = data_dir1 / f"retrosynthesis_{split}_rest.json"
        with open(rest_retro_file, 'w', encoding='utf-8') as f:
            json.dump([item['original_data'] for item in rest_retro],
                      f, indent=2, ensure_ascii=False)

        print(f"Saved {len(rest_retro)} unmatched retro records to {rest_retro_file}")

        # 更新剩余的机制数据
        mechanism_preprocessed = rest_mechanism

    # 保存剩余的机制数据
    rest_mechanism_file = data_dir1 / "mech-USPTO_rest.json"
    with open(rest_mechanism_file, 'w', encoding='utf-8') as f:
        json.dump([item['original_data'] for item in mechanism_preprocessed],
                  f, indent=2, ensure_ascii=False)

    print(f"Saved {len(mechanism_preprocessed)} unmatched mechanism records to {rest_mechanism_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process chemical mechanism and retrosynthesis data')
    parser.add_argument('--data_dir1', type=str, default='/mnt/e/DataSets/Chemistry/ChemechProcessing/',
                        help='Directory for mechanism data')
    parser.add_argument('--data_dir2', type=str, default='/mnt/e/DataSets/Chemistry/RetroSynthesis',
                        help='Directory for retrosynthesis data')
    parser.add_argument('--data_file', type=str, default='mech-USPTO-31k_train0.json',
                        help='Mechanism data file name')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Split names to process')

    args = parser.parse_args()

    main(args.data_dir1, args.data_dir2, args.data_file, args.splits)