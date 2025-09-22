# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: convert.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/22 18:32
# https://chat.deepseek.com/a/chat/s/8c3a0d52-01f4-4550-8e76-53dfe0753014
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors


def analyze_molecule(smiles):
    """分析分子并返回详细信息"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 计算分子式
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

    # 计算分子量
    molecular_weight = Descriptors.MolWt(mol)

    # 获取其他信息
    exact_mass = Descriptors.ExactMolWt(mol)
    heavy_atom_count = Descriptors.HeavyAtomCount(mol)

    return {
        'smiles': smiles,
        'formula': formula,
        'molecular_weight': molecular_weight,
        'exact_mass': exact_mass,
        'heavy_atom_count': heavy_atom_count
    }


# 测试多个分子
test_smiles = [
    "CCO",  # 乙醇
    "C1=CC=CC=C1",  # 苯
    "CC(=O)O",  # 乙酸
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
    "CCOC(=O)C"  # 乙酸乙酯
]

for smiles in test_smiles:
    result = analyze_molecule(smiles)
    if result:
        print(f"SMILES: {result['smiles']}")
        print(f"分子式: {result['formula']}")
        print(f"分子量: {result['molecular_weight']:.2f}")
        print(f"精确质量: {result['exact_mass']:.4f}")
        print("-" * 40)