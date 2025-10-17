# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: check_reaction_identical.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/8 13:52
# https://chat.deepseek.com/a/chat/s/ef46e810-9cb3-423d-87d6-eab301d29064

from rdkit import Chem


def compare_atom_mapped_reactions(predicted_rxn, ground_truth_rxn, verbose=False):
    """
    比较两个原子映射的SMILES反应表达式是否代表相同的反应

    参数:
    predicted_rxn: 预测的反应SMILES字符串
    ground_truth_rxn: 真实反应SMILES字符串
    verbose: 是否输出详细比较信息

    返回:
    bool: 两个反应是否相同
    dict: 详细的比较结果信息
    """

    def parse_reaction_smiles(rxn_smiles):
        """解析反应SMILES字符串"""
        try:
            reactants, products = rxn_smiles.split('>>')
            reactant_mol = Chem.MolFromSmiles(reactants, sanitize=True)
            product_mol = Chem.MolFromSmiles(products, sanitize=True)

            if reactant_mol is None or product_mol is None:
                raise ValueError("无法解析SMILES字符串")

            return reactant_mol, product_mol
        except Exception as e:
            raise ValueError(f"解析反应SMILES失败: {e}")

    def get_atom_mapping_info(mol):
        """获取分子的原子映射信息"""
        mapping_dict = {}
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                map_num = int(atom.GetProp('molAtomMapNumber'))
                mapping_dict[map_num] = atom.GetIdx()
        return mapping_dict

    def get_bond_info(mol):
        """获取分子的键信息"""
        bonds = []
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_map = int(begin_atom.GetProp('molAtomMapNumber')) if begin_atom.HasProp('molAtomMapNumber') else None
            end_map = int(end_atom.GetProp('molAtomMapNumber')) if end_atom.HasProp('molAtomMapNumber') else None

            if begin_map is not None and end_map is not None:
                bonds.append({
                    'atoms': tuple(sorted([begin_map, end_map])),
                    'type': bond.GetBondType(),
                    'stereo': bond.GetStereo(),
                    'is_conjugated': bond.GetIsConjugated()
                })
        return bonds

    def get_atom_stereo_info(mol):
        """获取原子的立体化学信息"""
        stereo_info = {}
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                map_num = int(atom.GetProp('molAtomMapNumber'))
                stereo_info[map_num] = {
                    'chiral_tag': atom.GetChiralTag(),
                    'hybridization': atom.GetHybridization()
                }
        return stereo_info

    def get_bond_stereo_info(mol):
        """获取键的立体化学信息"""
        stereo_bonds = {}
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            if begin_atom.HasProp('molAtomMapNumber') and end_atom.HasProp('molAtomMapNumber'):
                begin_map = int(begin_atom.GetProp('molAtomMapNumber'))
                end_map = int(end_atom.GetProp('molAtomMapNumber'))
                bond_key = tuple(sorted([begin_map, end_map]))

                stereo_bonds[bond_key] = {
                    'stereo': bond.GetStereo(),
                    'double_bond_stereo': bond.GetStereoAtoms() if bond.GetBondType() == Chem.BondType.DOUBLE else None
                }
        return stereo_bonds

    try:
        # 解析反应
        pred_reactants, pred_products = parse_reaction_smiles(predicted_rxn)
        gt_reactants, gt_products = parse_reaction_smiles(ground_truth_rxn)

        # 获取原子映射信息
        pred_reactant_mapping = get_atom_mapping_info(pred_reactants)
        pred_product_mapping = get_atom_mapping_info(pred_products)
        gt_reactant_mapping = get_atom_mapping_info(gt_reactants)
        gt_product_mapping = get_atom_mapping_info(gt_products)

        # 比较反应物和产物的原子映射集合
        pred_reactant_atoms = set(pred_reactant_mapping.keys())
        pred_product_atoms = set(pred_product_mapping.keys())
        gt_reactant_atoms = set(gt_reactant_mapping.keys())
        gt_product_atoms = set(gt_product_mapping.keys())

        # 检查原子映射是否匹配
        mapping_match = (pred_reactant_atoms == gt_reactant_atoms and
                         pred_product_atoms == gt_product_atoms)

        if not mapping_match:
            if verbose:
                print("原子映射不匹配")
                print(f"预测反应物原子: {pred_reactant_atoms}")
                print(f"真实反应物原子: {gt_reactant_atoms}")
                print(f"预测产物原子: {pred_product_atoms}")
                print(f"真实产物原子: {gt_product_atoms}")
            return False, {"reason": "原子映射不匹配"}

        # 比较反应物键信息
        pred_reactant_bonds = get_bond_info(pred_reactants)
        gt_reactant_bonds = get_bond_info(gt_reactants)

        # 比较产物键信息
        pred_product_bonds = get_bond_info(pred_products)
        gt_product_bonds = get_bond_info(gt_products)

        # 比较键信息
        def compare_bond_lists(bonds1, bonds2, label):
            bonds1_sorted = sorted(bonds1, key=lambda x: x['atoms'])
            bonds2_sorted = sorted(bonds2, key=lambda x: x['atoms'])

            if len(bonds1_sorted) != len(bonds2_sorted):
                if verbose:
                    print(f"{label}键数量不匹配: {len(bonds1_sorted)} vs {len(bonds2_sorted)}")
                return False

            for b1, b2 in zip(bonds1_sorted, bonds2_sorted):
                if b1['atoms'] != b2['atoms'] or b1['type'] != b2['type']:
                    if verbose:
                        print(f"{label}键不匹配: {b1} vs {b2}")
                    return False
            return True

        # 检查反应物和产物的键是否匹配
        reactants_bonds_match = compare_bond_lists(pred_reactant_bonds, gt_reactant_bonds, "反应物")
        products_bonds_match = compare_bond_lists(pred_product_bonds, gt_product_bonds, "产物")

        if not reactants_bonds_match or not products_bonds_match:
            return False, {"reason": "化学键不匹配"}

        # 比较立体化学信息
        pred_reactant_stereo = get_atom_stereo_info(pred_reactants)
        gt_reactant_stereo = get_atom_stereo_info(gt_reactants)
        pred_product_stereo = get_atom_stereo_info(pred_products)
        gt_product_stereo = get_atom_stereo_info(gt_products)

        # 比较原子立体化学
        def compare_atom_stereo(stereo1, stereo2, label):
            for map_num in stereo1:
                if (stereo1[map_num]['chiral_tag'] != stereo2[map_num]['chiral_tag'] or
                        stereo1[map_num]['hybridization'] != stereo2[map_num]['hybridization']):
                    if verbose:
                        print(f"{label}原子{map_num}立体化学不匹配")
                    return False
            return True

        atom_stereo_match = (compare_atom_stereo(pred_reactant_stereo, gt_reactant_stereo, "反应物") and
                             compare_atom_stereo(pred_product_stereo, gt_product_stereo, "产物"))

        if not atom_stereo_match:
            return False, {"reason": "原子立体化学不匹配"}

        # 比较键立体化学
        pred_reactant_bond_stereo = get_bond_stereo_info(pred_reactants)
        gt_reactant_bond_stereo = get_bond_stereo_info(gt_reactants)
        pred_product_bond_stereo = get_bond_stereo_info(pred_products)
        gt_product_bond_stereo = get_bond_stereo_info(gt_products)

        def compare_bond_stereo(stereo1, stereo2, label):
            for bond_key in stereo1:
                if (stereo1[bond_key]['stereo'] != stereo2[bond_key]['stereo']):
                    if verbose:
                        print(f"{label}键{bond_key}立体化学不匹配")
                    return False
            return True

        bond_stereo_match = (compare_bond_stereo(pred_reactant_bond_stereo, gt_reactant_bond_stereo, "反应物") and
                             compare_bond_stereo(pred_product_bond_stereo, gt_product_bond_stereo, "产物"))

        if not bond_stereo_match:
            return False, {"reason": "键立体化学不匹配"}

        # 所有检查通过
        if verbose:
            print("两个反应完全相同")
        return True, {"reason": "所有检查通过"}

    except Exception as e:
        if verbose:
            print(f"比较过程中发生错误: {e}")
        return False, {"reason": f"解析错误: {e}"}


# 测试示例
def test_comparison():
    """测试比较函数"""

    # 示例1: 相同的反应
    rxn1 = "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])=[O:7]"
    rxn2 = "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])=[O:7]"

    print("示例1: 相同的反应")
    result, info = compare_atom_mapped_reactions(rxn1, rxn2, verbose=True)
    print(f"结果: {result}, 原因: {info['reason']}\n")

    # 示例2: 原子映射不同但化学相同的反应
    rxn3 = "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])=[O:7]"
    rxn4 = "[CH3:10][C:20](=[O:30])[OH:40]>>[CH3:10][C:20](=[O:30])[O:40][C:50]([CH3:60])=[O:70]"

    print("示例2: 原子映射编号不同但化学相同的反应")
    result, info = compare_atom_mapped_reactions(rxn3, rxn4, verbose=True)
    print(f"结果: {result}, 原因: {info['reason']}\n")

    # 示例3: 键类型不同的反应
    rxn5 = "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])=[O:7]"
    rxn6 = "[CH3:1][C:2](#[O:3])[OH:4]>>[CH3:1][C:2](#[O:3])[O:4][C:5]([CH3:6])=[O:7]"  # 双键变三键

    print("示例3: 键类型不同的反应")
    result, info = compare_atom_mapped_reactions(rxn5, rxn6, verbose=True)
    print(f"结果: {result}, 原因: {info['reason']}\n")

    # 示例4: 立体化学不同的反应
    rxn7 = "C[C@@H](O)[C:1]=[O:2]>>C[C@@H](O)[C:1]=[O:2]"  # 手性中心
    rxn8 = "C[C@H](O)[C:1]=[O:2]>>C[C@H](O)[C:1]=[O:2]"  # 相反的手性

    print("示例4: 立体化学不同的反应")
    result, info = compare_atom_mapped_reactions(rxn7, rxn8, verbose=True)
    print(f"结果: {result}, 原因: {info['reason']}\n")

    # 示例5: 原子缺失的反应
    rxn9 = "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])=[O:7]"
    rxn10 = "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O:4][C:5]=[O:7]"  # 缺少CH3:6

    print("示例5: 原子缺失的反应")
    result, info = compare_atom_mapped_reactions(rxn9, rxn10, verbose=True)
    print(f"结果: {result}, 原因: {info['reason']}\n")


if __name__ == "__main__":
    test_comparison()
