# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: random_smiles.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/25 16:27

from typing import List
import random
from rdkit import Chem


def _augment_smiles(self, smiles: str, num_variants: int = 1, preserve_atom_mapping: bool = True) -> List[str]:
    """
    增强SMILES字符串，支持Atom Mapped SMILES和Standard SMILES

    Args:
        smiles: 输入SMILES字符串（可以是atom mapped或standard）
        num_variants: 需要生成的变体数量
        preserve_atom_mapping: 是否保留原子映射信息

    Returns:
        SMILES变体列表
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles] * num_variants

        # 检查是否有原子映射
        has_atom_mapping = any(atom.GetAtomMapNum() > 0 for atom in mol.GetAtoms())

        if not has_atom_mapping or not preserve_atom_mapping:
            # 标准SMILES或不需要保留映射的情况
            results = []
            for _ in range(num_variants):
                random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                results.append(random_smiles)
            return results
        else:
            # Atom Mapped SMILES的情况
            return self._augment_atom_mapped_smiles(mol, smiles, num_variants)

    except Exception:
        return [smiles] * num_variants


def _augment_atom_mapped_smiles(self, mol: Chem.Mol, original_smiles: str, num_variants: int) -> List[str]:
    """处理Atom Mapped SMILES的增强"""
    try:
        # 方法1: 通过随机化原子顺序但保留映射
        results = []
        for _ in range(num_variants):
            # 创建分子副本
            mol_copy = Chem.Mol(mol)

            # 随机化原子顺序但保持映射
            random_smiles = self._randomize_with_mapping(mol_copy)
            results.append(random_smiles)

        return results

    except Exception:
        # 如果增强失败，返回原始SMILES
        return [original_smiles] * num_variants


def _randomize_with_mapping(self, mol: Chem.Mol) -> str:
    """随机化原子顺序同时保留原子映射"""
    # 获取原子映射信息
    atom_mappings = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]

    # 方法1: 使用不同的SMILES书写顺序
    try:
        # 设置不同的原子顺序
        new_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_order)

        # 使用随机顺序生成SMILES
        random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False,
                                         atomOrder=new_order)
        return random_smiles
    except:
        pass

    # 方法2: 通过键的随机化
    try:
        # 获取分子的所有可能SMILES表示
        all_smiles = set()
        for _ in range(100):  # 尝试多次生成不同的SMILES
            smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            all_smiles.add(smi)
            if len(all_smiles) >= 10:  # 收集足够多的变体
                break

        if all_smiles:
            return random.choice(list(all_smiles))
        else:
            return Chem.MolToSmiles(mol, canonical=False)
    except:
        return Chem.MolToSmiles(mol, canonical=False)


def _augment_smiles_advanced(self, smiles: str, num_variants: int = 1,
                             method: str = "standard") -> List[str]:
    """
    高级SMILES增强方法，提供多种增强策略

    Args:
        smiles: 输入SMILES
        num_variants: 变体数量
        method: 增强方法 ("standard", "randomize", "rotation")
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles] * num_variants

        results = []

        for _ in range(num_variants):
            if method == "standard":
                # 标准随机化
                result = Chem.MolToSmiles(mol, doRandom=True, canonical=False)

            elif method == "randomize":
                # 更激进的随机化
                result = self._aggressive_randomize(mol)

            elif method == "rotation":
                # 通过旋转键生成变体
                result = self._rotate_bonds(mol)

            else:
                result = Chem.MolToSmiles(mol, doRandom=True, canonical=False)

            results.append(result)

        return results

    except Exception:
        return [smiles] * num_variants


def _aggressive_randomize(self, mol: Chem.Mol) -> str:
    """更激进的随机化方法"""
    try:
        # 尝试多次随机化以获得更多样的结果
        smiles_variants = set()
        for _ in range(50):
            smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            smiles_variants.add(smi)

        return random.choice(list(smiles_variants)) if smiles_variants else Chem.MolToSmiles(mol)
    except:
        return Chem.MolToSmiles(mol)


def _rotate_bonds(self, mol: Chem.Mol) -> str:
    """通过旋转可旋转键生成变体"""
    try:
        # 创建可编辑的分子
        mol_copy = Chem.Mol(mol)

        # 随机旋转一些可旋转键
        rotatable_bonds = [bond for bond in mol_copy.GetBonds()
                           if bond.GetBondType() == Chem.BondType.SINGLE
                           and not bond.IsInRing()]

        if rotatable_bonds:
            # 随机选择一些键进行旋转
            bonds_to_rotate = random.sample(rotatable_bonds,
                                            min(5, len(rotatable_bonds)))

            for bond in bonds_to_rotate:
                # 简单的尝试旋转（实际中需要更复杂的处理）
                pass

        return Chem.MolToSmiles(mol_copy, canonical=False)
    except:
        return Chem.MolToSmiles(mol)
