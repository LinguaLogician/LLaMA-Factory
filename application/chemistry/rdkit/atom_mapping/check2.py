# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: check2.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/8 16:31
# https://chat.deepseek.com/a/chat/s/58587870-15a3-405d-83f5-5de9c8058ba2

from rdkit import Chem
from itertools import permutations

def parse_reaction(reaction_smi: str):
    """解析反应式为反应物和生成物分子列表"""
    reactants_smi, products_smi = reaction_smi.split(">>")
    reactants = [Chem.MolFromSmiles(s) for s in reactants_smi.split(".") if s]
    products = [Chem.MolFromSmiles(s) for s in products_smi.split(".") if s]
    return reactants, products

def mol_signature(mol):
    """
    根据分子生成“结构签名”，忽略映射号但保留连接信息。
    这个签名用于判断结构是否一致。
    """
    if mol is None:
        return None

    atoms_info = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        neighbors = sorted([nbr.GetSymbol() for nbr in atom.GetNeighbors()])
        atoms_info.append((symbol, tuple(neighbors)))
    return sorted(atoms_info)

def compare_mol_sets(pred_mols, gt_mols):
    """
    判断两个分子集合是否等价（忽略映射号和顺序）
    """
    if len(pred_mols) != len(gt_mols):
        return False

    gt_signatures = [mol_signature(m) for m in gt_mols]
    pred_signatures = [mol_signature(m) for m in pred_mols]

    # 尝试所有排列组合以防分子顺序不同
    for perm in permutations(pred_signatures):
        if all(p == g for p, g in zip(perm, gt_signatures)):
            return True
    return False

def is_same_reaction(predicted_rxn: str, ground_truth_rxn: str) -> bool:
    """
    判断两个原子映射反应是否为同一反应（忽略编号差异）
    """
    try:
        pred_reactants, pred_products = parse_reaction(predicted_rxn)
        gt_reactants, gt_products = parse_reaction(ground_truth_rxn)
    except Exception:
        return False

    # 检查反应物与生成物集合是否等价
    same_reactants = compare_mol_sets(pred_reactants, gt_reactants)
    same_products = compare_mol_sets(pred_products, gt_products)

    return same_reactants and same_products


# ---------------- 示例 ----------------
if __name__ == "__main__":
    # 示例1：相同反应，但映射编号不同
    # predicted_rxn = "[CH3:1][CH2:2][OH:3]>>[CH3:5][CH:6]=[O:7]"
    # ground_truth_rxn = "[CH3:11][CH2:12][OH:13]>>[CH3:21][CH:22]=[O:23]"
    # print(is_same_reaction(predicted_rxn, ground_truth_rxn))
    # # ✅ 输出: True
    #
    # # 示例2：不同反应（结构不同）
    # predicted_rxn2 = "[CH3:1][CH2:2][OH:3]>>[CH3:5][CH:6]=[O:7]"
    # ground_truth_rxn2 = "[CH3:1][CH2:2][Cl:3]>>[CH3:5][CH:6]=[O:7]"
    # print(is_same_reaction(predicted_rxn2, ground_truth_rxn2))
    # # ✅ 输出: False
    #
    # # 示例3：多个分子反应（含"."）
    # predicted_rxn3 = "[CH3:1][OH:2].[Na:3]>>[CH3:4][ONa:5]"
    # ground_truth_rxn3 = "[CH3:10][OH:11].[Na:12]>>[CH3:13][ONa:14]"
    # print(is_same_reaction(predicted_rxn3, ground_truth_rxn3))
    # # ✅ 输出: True

    # predicted_rxn3 = "CC(C)(C)OC(=O)[NH:2][CH2:3][C:4](=[O:5])[N:6]1[CH2:7][CH2:8][c:9]2[c:10]([Br:11])[cH:12][cH:13][cH:14][c:15]2[CH:16]1[CH2:17][C:18](=[O:19])[OH:20]>>[NH3+:2][CH2:3][C:4](=[O:5])[N:6]1[CH2:7][CH2:8][c:9]2[c:10]([Br:11])[cH:12][cH:13][cH:14][c:15]2[CH:16]1[CH2:17][C:18](=[O:19])[OH:20]"
    # predicted_rxn3 = "CC(C)(C)OC(=O)[NH:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:10])[cH:11][cH:12][cH:13][c:14]2[CH:15]1[CH2:16][C:17](=[O:18])[OH:19]>>[NH2:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:10])[cH:11][cH:12][cH:13][c:14]2[CH:15]1[CH2:16][C:17](=[O:18])[OH:19]"
    # predicted_rxn3 = "CC(C)(C)O[C:1](=[O:2])[NH:3][CH2:4][C:5](=[O:6])[N:7]1[CH2:8][CH2:9][c:10]2[c:11]([Br:12])[cH:13][cH:14][cH:15][c:16]2[CH:17]1[CH2:18][C:19](=[O:20])[OH:21]>>[NH3+:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:12])[cH:13][cH:14][cH:15][c:16]2[CH:17]1[CH2:18][C:19](=[O:20])[OH:21]"
    # predicted_rxn3 = "[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH2:7][CH3:8])[c:9]1[N:10]=[C:11]=[O:12].[cH:13]1[cH:14][cH:15][c:16]([CH:17]2[CH2:18][CH2:19][CH:20]([NH:21][CH2:22][CH:23]3[CH2:24][CH2:25][CH2:26][CH2:27][CH2:28]3)[CH2:29][CH2:30]2)[cH:31][cH:32]1>>[CH3:1][c:2]1[cH:3][cH:4][cH:5][c:6]([CH2:7][CH3:8])[c:9]1[NH:10][C:11](=[O:12])[N:21]([CH:20]1[CH2:19][CH2:18][CH:17]([c:16]2[cH:15][cH:14][cH:13][cH:32][cH:31]2)[CH2:30][CH2:29]1)[CH2:22][CH:23]1[CH2:24][CH2:25][CH2:26][CH2:27][CH2:28]1"
    # ground_truth_rxn3 = "[CH3:1][CH2:2][c:3]1[cH:4][cH:5][cH:6][c:7]([CH2:8][CH3:9])[c:10]1[N:11]=[C:12]=[O:13].[cH:14]1[cH:15][cH:16][c:17]([CH:18]2[CH2:19][CH2:20][CH:21]([NH:22][CH2:23][CH:24]3[CH2:25][CH2:26][CH2:27][CH2:28][CH2:29]3)[CH2:30][CH2:31]2)[cH:32][cH:33]1>>[CH3:1][CH2:2][c:3]1[cH:4][cH:5][cH:6][c:7]([CH2:8][CH3:9])[c:10]1[NH:11][C:12](=[O:13])[N:22]([CH:21]1[CH2:20][CH2:19][CH:18]([c:17]2[cH:16][cH:15][cH:14][cH:33][cH:32]2)[CH2:31][CH2:30]1)[CH2:23][CH:24]1[CH2:25][CH2:26][CH2:27][CH2:28][CH2:29]1"
    ground_truth_rxn3 = "[Na+].[CH2:1]([CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][S:7][CH2:8][CH2:9]1)[Cl:101].[CH3:10][N:11]([CH3:12])[CH2:13][C:14]1([c:15]2[cH:16][cH:17][c:18]([OH:19])[cH:20][cH:21]2)[CH2:22][CH2:23][O:24][CH2:25][CH2:26]1.[H-:301]>>[CH2:1]([CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][S:7][CH2:8][CH2:9]1)[O:19][c:18]1[cH:17][cH:16][c:15]([C:14]2([CH2:13][N:11]([CH3:10])[CH3:12])[CH2:22][CH2:23][O:24][CH2:25][CH2:26]2)[cH:21][cH:20]1"

    predicted_rxn3 = "[Na+].[CH2:1]([CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][S:7][CH2:8][CH2:9]1)[Cl:101].[CH3:10][N:11]([CH3:12])[CH2:13][C:14]1([c:15]2[cH:16][cH:17][c:18]([OH:19])[cH:20][cH:21]2)[CH2:22][CH2:23][O:24][CH2:25][CH2:26]1.[H-:301]>>[CH2:1]([CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][S:7][CH2:8][CH2:9]1)[O:19][c:18]1[cH:17][cH:16][c:15](/[C:14]2([CH2:13][N:11]([CH3:10])[CH3:12])[CH2:22][CH2:23][O:24][CH2:25][CH2:26]2)[cH:21][cH:20]1"
    print("Hello: " + str(is_same_reaction(predicted_rxn3, ground_truth_rxn3)))
    # ✅ 输出: True
