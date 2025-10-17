# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: compare.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/8 14:13
# https://chat.deepseek.com/a/chat/s/5a0a745b-ae7d-4f62-a5fb-92a96ecf1a3f
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Tuple, List




def compare_reactions(predicted_rxn: str, ground_truth_rxn: str) -> bool:
    def canonicalize_smiles_list(smiles_list: List[str]) -> str:
        """
        将SMILES列表进行标准化并排序组合

        Args:
            smiles_list: SMILES字符串列表

        Returns:
            标准化并排序后的组合SMILES字符串
        """
        canonical_smiles = []

        for smiles in smiles_list:
            try:
                # 从SMILES创建分子对象
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 生成标准化的SMILES（移除原子映射信息）
                    for atom in mol.GetAtoms():
                        atom.SetAtomMapNum(0)
                    canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            except:
                # 如果解析失败，保留原SMILES
                canonical_smiles.append(smiles)

        # 按字母顺序排序并合并
        return '.'.join(sorted(canonical_smiles))

    def preprocess_reaction_smiles(reaction_smiles: str) -> Tuple[str, str]:
        """
        预处理反应SMILES，分离反应物和生成物并进行标准化

        Args:
            reaction_smiles: 反应SMILES字符串

        Returns:
            (标准化反应物, 标准化生成物)
        """
        # 分割反应物和生成物
        if '>>' not in reaction_smiles:
            raise ValueError(f"无效的反应SMILES格式: {reaction_smiles}")

        reactants_str, products_str = reaction_smiles.split('>>')

        # 分割多个分子
        reactants_list = reactants_str.split('.')
        products_list = products_str.split('.')

        # 标准化并重新组合
        canonical_reactants = canonicalize_smiles_list(reactants_list)
        canonical_products = canonicalize_smiles_list(products_list)

        return canonical_reactants, canonical_products

    """
    比较两个Atom-mapped SMILES反应是否相同

    Args:
        predicted_rxn: 预测的反应SMILES
        ground_truth_rxn: 真实反应SMILES

    Returns:
        bool: 两个反应是否相同
    """
    try:
        # 预处理两个反应
        pred_reactants, pred_products = preprocess_reaction_smiles(predicted_rxn)
        true_reactants, true_products = preprocess_reaction_smiles(ground_truth_rxn)

        # 比较反应物和生成物
        reactants_match = pred_reactants == true_reactants
        products_match = pred_products == true_products

        return reactants_match and products_match

    except Exception as e:
        print(f"处理反应时发生错误: {e}")
        return False


# 测试示例
if __name__ == "__main__":
    # 示例1: 相同的反应（原子映射不同但化学本质相同）
    predicted_rxn1 = "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])=[O:7]"
    ground_truth_rxn1 = "[CH3:10][C:20](=[O:30])[OH:40]>>[CH3:10][C:20](=[O:30])[O:40][C:50]([CH3:60])=[O:70]"

    result1 = compare_reactions(predicted_rxn1, ground_truth_rxn1)
    print(f"示例1 - 反应是否相同: {result1}")

    # 示例2: 不同的反应物
    predicted_rxn2 = "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])=[O:7]"
    ground_truth_rxn2 = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][O:3][C:4]([CH3:5])=[O:6]"

    result2 = compare_reactions(predicted_rxn2, ground_truth_rxn2)
    print(f"示例2 - 反应是否相同: {result2}")

    # 示例3: 相同的反应但分子顺序不同
    predicted_rxn3 = "[CH3:1]O.[C:2]=[O:3]>>[CH3:1][O:2][C:3]=[O:4]"
    ground_truth_rxn3 = "[C:2]=[O:3].[CH3:1]O>>[CH3:1][O:2][C:3]=[O:4]"

    result3 = compare_reactions(predicted_rxn3, ground_truth_rxn3)
    print(f"示例3 - 反应是否相同: {result3}")

    # 示例4: 多个分子的情况
    predicted_rxn4 = "CCO.CC(=O)O>>CCOC(=O)C.O"
    ground_truth_rxn4 = "CC(=O)O.CCO>>CCOC(=O)C.O"

    result4 = compare_reactions(predicted_rxn4, ground_truth_rxn4)
    print(f"示例4 - 反应是否相同: {result4}")

    # 打印详细处理过程
    print("\n详细处理过程:")
    for i, (pred, true) in enumerate([(predicted_rxn1, ground_truth_rxn1),
                                      (predicted_rxn3, ground_truth_rxn3)], 1):
        pred_r, pred_p = preprocess_reaction_smiles(pred)
        true_r, true_p = preprocess_reaction_smiles(true)
        print(f"示例{i}:")
        print(f"  预测反应物: {pred_r}")
        print(f"  真实反应物: {true_r}")
        print(f"  预测生成物: {pred_p}")
        print(f"  真实生成物: {true_p}")
        print(f"  反应物匹配: {pred_r == true_r}")
        print(f"  生成物匹配: {pred_p == true_p}")