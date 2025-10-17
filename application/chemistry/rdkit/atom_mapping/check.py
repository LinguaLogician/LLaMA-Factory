# https://chat.deepseek.com/a/chat/s/0910c380-a2da-48da-9f25-e986b07f61a2

from collections import defaultdict
import re


def are_reactions_identical(predicted_rxn, ground_truth_rxn):
    """
    判断两个原子映射的SMILES反应是否相同

    参数:
        predicted_rxn: 预测的反应表达式 (e.g., "[CH3:1][OH:2]>>[CH2:1]=[OH:2]")
        ground_truth_rxn: 真实反应表达式

    返回:
        bool: 如果两个反应相同返回True，否则返回False
    """

    def parse_reaction_smiles(rxn_smiles):
        """解析反应SMILES，提取反应物、产物和原子映射"""
        try:
            reactants, products = rxn_smiles.split('>>')
        except ValueError:
            raise ValueError(f"无效的反应格式: {rxn_smiles}")

        # 解析反应物和产物中的原子映射
        reactant_mappings = extract_atom_mappings(reactants)
        product_mappings = extract_atom_mappings(products)

        return reactants, products, reactant_mappings, product_mappings

    def extract_atom_mappings(smiles_part):
        """从SMILES部分提取原子映射信息"""
        # 使用正则表达式匹配原子映射，如 [C:1], [CH3:2], [N:10] 等
        pattern = r'\[([^]]*):(\d+)\]'
        matches = re.findall(pattern, smiles_part)

        # 创建映射字典：原子编号 -> 原子符号
        mappings = {}
        for atom_symbol, mapping_num in matches:
            mappings[int(mapping_num)] = atom_symbol

        return mappings

    def extract_bond_information(smiles_part, atom_mappings):
        """从SMILES中提取键信息（考虑原子映射）"""
        # 移除映射标记以便分析连接性
        clean_smiles = re.sub(r':(\d+)', '', smiles_part)

        # 提取原子序列（保持顺序）
        atoms = re.findall(r'\[([^]]*)\]', clean_smiles)

        # 构建连接性图
        bond_graph = defaultdict(list)
        stack = []
        current_atom_idx = -1

        # 简化的SMILES解析来获取连接性
        i = 0
        atom_idx = 0
        while i < len(clean_smiles):
            char = clean_smiles[i]

            if char == '[':
                # 开始原子
                j = i + 1
                while j < len(clean_smiles) and clean_smiles[j] != ']':
                    j += 1
                atom_symbol = clean_smiles[i + 1:j]

                current_atom_idx = atom_idx
                atom_idx += 1
                i = j

            elif char in ('=', '#', ':'):
                # 键类型，暂时跳过具体处理
                pass
            elif char == '(':
                # 开始分支
                stack.append(current_atom_idx)
            elif char == ')':
                # 结束分支
                if stack:
                    current_atom_idx = stack.pop()
            elif char.isalnum() and char not in ('=', '#', ':'):
                # 简单原子（无括号），在实际应用中需要更复杂的处理
                pass

            i += 1

        # 这里简化处理，实际应用中需要更完整的SMILES解析
        # 返回原子序列作为简化的结构表示
        return atoms

    def create_reaction_signature(reactant_mappings, product_mappings):
        """
        创建反应的签名，基于映射关系模式而不是具体编号
        """
        # 创建反应物到产物的映射关系模式
        mapping_relationship = {}

        # 检查所有在反应物和产物中都出现的原子
        common_atoms = set(reactant_mappings.keys()) & set(product_mappings.keys())

        for atom_id in common_atoms:
            reactant_atom = reactant_mappings[atom_id]
            product_atom = product_mappings[atom_id]
            mapping_relationship[atom_id] = (reactant_atom, product_atom)

        # 创建规范化的签名：对映射关系进行排序，忽略具体的原子编号
        sorted_relationships = sorted(mapping_relationship.items(),
                                      key=lambda x: (x[1][0], x[1][1]))

        # 签名只包含原子类型的变化关系
        signature = tuple((reactant_type, product_type)
                          for _, (reactant_type, product_type) in sorted_relationships)

        return signature

    try:
        # 解析两个反应
        pred_reactants, pred_products, pred_reactant_mappings, pred_product_mappings = parse_reaction_smiles(
            predicted_rxn)
        gt_reactants, gt_products, gt_reactant_mappings, gt_product_mappings = parse_reaction_smiles(ground_truth_rxn)

        # 创建反应签名
        pred_signature = create_reaction_signature(pred_reactant_mappings, pred_product_mappings)
        gt_signature = create_reaction_signature(gt_reactant_mappings, gt_product_mappings)

        # 比较签名
        return pred_signature == gt_signature

    except Exception as e:
        print(f"错误处理反应: {e}")
        return False


def are_reactions_identical_advanced(predicted_rxn, ground_truth_rxn):
    """
    更高级的版本，考虑分子图和反应中心
    """

    def create_mapping_graph(rxn_smiles):
        """创建基于原子映射的图表示"""
        reactants, products = rxn_smiles.split('>>')

        # 提取所有映射
        pattern = r'\[([^]]*):(\d+)\]'

        reactant_matches = re.findall(pattern, reactants)
        product_matches = re.findall(pattern, products)

        reactant_map = {int(num): symbol for symbol, num in reactant_matches}
        product_map = {int(num): symbol for symbol, num in product_matches}

        # 找出变化的原子（在反应物和产物中类型不同的原子）
        changing_atoms = []
        for atom_id in set(reactant_map.keys()) & set(product_map.keys()):
            if reactant_map[atom_id] != product_map[atom_id]:
                changing_atoms.append(atom_id)

        # 创建反应中心签名
        reaction_center = tuple(sorted(changing_atoms))

        # 创建原子类型变化模式
        change_pattern = []
        for atom_id in sorted(set(reactant_map.keys()) | set(product_map.keys())):
            reactant_type = reactant_map.get(atom_id, 'None')
            product_type = product_map.get(atom_id, 'None')
            change_pattern.append((reactant_type, product_type))

        return tuple(change_pattern), reaction_center

    try:
        pred_pattern, pred_center = create_mapping_graph(predicted_rxn)
        gt_pattern, gt_center = create_mapping_graph(ground_truth_rxn)

        # 比较变化模式和反应中心
        return pred_pattern == gt_pattern and pred_center == gt_center

    except Exception as e:
        print(f"高级方法错误: {e}")
        return False


# 测试示例
if __name__ == "__main__":
    # 示例1: 相同的反应，不同的原子编号
    rxn1 = "[CH3:1][OH:2]>>[CH2:1]=[OH:2]"  # 醇脱水
    rxn2 = "[CH3:3][OH:4]>>[CH2:3]=[OH:4]"  # 相同的反应，不同编号

    print("示例1 - 相同的反应，不同的原子编号:")
    print(f"反应1: {rxn1}")
    print(f"反应2: {rxn2}")
    print(f"基础方法: {are_reactions_identical(rxn1, rxn2)}")
    print(f"高级方法: {are_reactions_identical_advanced(rxn1, rxn2)}")
    print()

    # 示例2: 不同的反应
    rxn3 = "[CH3:1][OH:2]>>[CH2:1]=[OH:2]"  # 醇脱水
    rxn4 = "[CH3:1][Cl:2]>>[CH2:1]=[OH:2]"  # 不同的反应物

    print("示例2 - 不同的反应:")
    print(f"反应3: {rxn3}")
    print(f"反应4: {rxn4}")
    print(f"基础方法: {are_reactions_identical(rxn3, rxn4)}")
    print(f"高级方法: {are_reactions_identical_advanced(rxn3, rxn4)}")
    print()

    # 示例3: 更复杂的反应
    rxn5 = "[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[Cl:3]"  # 羧酸转酰氯
    rxn6 = "[C:4](=[O:5])[OH:6]>>[C:4](=[O:5])[Cl:6]"  # 相同的反应

    print("示例3 - 复杂的相同反应:")
    print(f"反应5: {rxn5}")
    print(f"反应6: {rxn6}")
    print(f"基础方法: {are_reactions_identical(rxn5, rxn6)}")
    print(f"高级方法: {are_reactions_identical_advanced(rxn5, rxn6)}")
    print()

    # 示例4: 原子映射模式不同的反应
    rxn7 = "[CH3:1][OH:2]>>[CH2:1]=[OH:2]"  # 正常的脱水
    rxn8 = "[CH3:1][OH:2]>>[CH3:2][OH:1]"  # 原子交换，不同的反应

    print("示例4 - 原子映射模式不同:")
    print(f"反应7: {rxn7}")
    print(f"反应8: {rxn8}")
    print(f"基础方法: {are_reactions_identical(rxn7, rxn8)}")
    print(f"高级方法: {are_reactions_identical_advanced(rxn7, rxn8)}")

    prediction = "CC(C)(C)O[C:1](=[O:2])[NH:3][CH2:4][C:5](=[O:6])[N:7]1[CH2:8][CH2:9][c:10]2[c:11]([Br:12])[cH:13][cH:14][cH:15][c:16]2[CH:17]1[CH2:18][C:19](=[O:20])[OH:21]>>[NH3+:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:12])[cH:13][cH:14][cH:15][c:16]2[CH:17]1[CH2:18][C:19](=[O:20])[OH:21]"
    ground_truth = "CC(C)(C)OC(=O)[NH:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:10])[cH:11][cH:12][cH:13][c:14]2[CH:15]1[CH2:16][C:17](=[O:18])[OH:19]>>[NH3+:1][CH2:2][C:3](=[O:4])[N:5]1[CH2:6][CH2:7][c:8]2[c:9]([Br:10])[cH:11][cH:12][cH:13][c:14]2[CH:15]1[CH2:16][C:17](=[O:18])[OH:19]"

    print("示例5:")
    print(f"反应7: {prediction}")
    print(f"反应8: {ground_truth}")
    print(f"基础方法: {are_reactions_identical(prediction, ground_truth)}")
    print(f"高级方法: {are_reactions_identical_advanced(prediction, ground_truth)}")