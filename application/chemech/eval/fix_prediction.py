# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: fix_prediction.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/8 1:22
# https://chat.deepseek.com/a/chat/s/1adc5565-a280-43d4-a738-33efef8e6d2d
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import logging
from itertools import permutations

# 配置常量
BASE_DIR = "/mnt/e/Results/chemechpred/prediction"
OUTPUT_DIR = "/mnt/e/Results/chemechpred/prediction_fixed"
PROCESS_FOLDERS = [
    "rxts_to_prds_plus",
    "rxts_to_prds",
    "rxn_to_rxn",
]
# PROCESS_FOLDERS = [
#     "demo"
# ]

# 任务映射
TASK_MAPPING = {
    "RXN_TO_RXN": [
        "ORI.CANO.AM.RXN->UPD.CANO.AM.RXN",
        "ORI.CANO.STD.RXN->UPD.CANO.STD.RXN",
        "ORI.CANO.STD.RXN->ORI.CANO.AM.RXN",
        "UPD.CANO.STD.RXN->UPD.CANO.AM.RXN",
        "ORI.CANO.AM.RXN->ORI.CANO.STD.RXN",
        "UPD.CANO.AM.RXN->UPD.CANO.STD.RXN",
    ],
    "RXTS_TO_PRDS": [
        "UPD.CANO.AM.RXTS->UPD.CANO.AM.PRDS",
        "UPD.CANO.STD.RXTS->UPD.CANO.STD.PRDS",
        "ORI.CANO.AM.RXTS->ORI.CANO.AM.PRDS",
        "ORI.CANO.STD.RXTS->ORI.CANO.STD.PRDS",
    ],
    "RXTS_TO_PRDS_PLUS": [
        "UPD.CANO.AM.RXTS->MECH+UPD.CANO.AM.PRDS",
        "UPD.CANO.AM.RXTS->CLS+UPD.CANO.AM.PRDS",
        "UPD.CANO.AM.RXTS->CLS+MECH+UPD.CANO.AM.PRDS",
        "UPD.CANO.AM.RXTS+MECH->UPD.CANO.AM.PRDS",
        "UPD.CANO.AM.RXTS+CLS->UPD.CANO.AM.PRDS",
        "UPD.CANO.AM.RXTS+CLS->MECH+UPD.CANO.AM.PRDS",
        "UPD.CANO.AM.RXTS+MECH->CLS+UPD.CANO.AM.PRDS",
        "UPD.CANO.AM.RXTS+MECH+CLS->UPD.CANO.AM.PRDS"
    ],
}

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import re
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem


def extract_atom_mapping(smiles: str) -> Dict[int, int]:
    """从SMILES字符串中提取atom mapping信息"""
    atom_mapping = {}
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                map_num = int(atom.GetProp('molAtomMapNumber'))
                atom_idx = atom.GetIdx()
                atom_mapping[map_num] = atom_idx
    return atom_mapping


def compare_atom_mapping_relationship2(predicted_smiles: str, ground_truth_smiles: str,
                                      reactants_smiles: str) -> bool:
    """
    比较atom mapping关系是否一致
    对于反应物->产物的mapping关系进行比较
    """
    try:
        # 提取反应物的atom mapping
        reactants_mapping = extract_atom_mapping(reactants_smiles)

        # 提取预测产物和真实产物的atom mapping
        pred_mapping = extract_atom_mapping(predicted_smiles)
        gt_mapping = extract_atom_mapping(ground_truth_smiles)

        # 比较mapping关系：反应物中的原子是否映射到产物中的相同原子
        mapping_relationship_consistent = True

        for map_num, reactant_atom_idx in reactants_mapping.items():
            # 检查在预测结果和真实结果中，这个映射号是否都指向产物中的原子
            pred_has_mapping = map_num in pred_mapping
            gt_has_mapping = map_num in gt_mapping

            if pred_has_mapping != gt_has_mapping:
                mapping_relationship_consistent = False
                break

        return mapping_relationship_consistent

    except Exception as e:
        logger.error(f"Atom mapping比较出错: {str(e)}")
        return False

def compare_atom_mapping_relationship(predicted_smiles: str, ground_truth_smiles: str,
                                      reactants_smiles: str) -> bool:
    """
    比较atom mapping关系是否一致
    对于反应物->产物的mapping关系进行比较
    """
    try:
        predicted_rxn = reactants_smiles.strip()+">>"+predicted_smiles.strip()
        ground_truth_rxn = reactants_smiles.strip()+">>"+ground_truth_smiles.strip()
        return compare_reaction_atom_mapping(predicted_rxn, ground_truth_rxn)

    except Exception as e:
        logger.error(f"Atom mapping比较出错: {str(e)}")
        return False


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


def compare_reaction_atom_mapping2(predicted_rxn: str, ground_truth_rxn: str) -> bool:
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
        if not compare_reactions(predicted_rxn, ground_truth_rxn):
            return False

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



def compare_reaction_atom_mapping(predicted_rxn: str, ground_truth_rxn: str) -> bool:
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
    """
    判断两个原子映射反应是否为同一反应（忽略编号差异）
    """
    try:
        if not compare_reactions(predicted_rxn, ground_truth_rxn):
            return False
        pred_reactants, pred_products = parse_reaction(predicted_rxn)
        gt_reactants, gt_products = parse_reaction(ground_truth_rxn)
    except Exception:
        return False

    # 检查反应物与生成物集合是否等价
    same_reactants = compare_mol_sets(pred_reactants, gt_reactants)
    same_products = compare_mol_sets(pred_products, gt_products)

    return same_reactants and same_products



class DataProcessor:
    def __init__(self, base_dir: str, output_dir: str, process_folders: List[str]):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.process_folders = process_folders
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_task_id(self, instruction: str) -> str:
        """将instruction转换为task_id格式"""
        task_id = instruction.replace("->", "_TO_").replace(".", "").replace("+", "_")
        return task_id.upper()

    def parse_input_output_components(self, text: str, task_type: str) -> Dict[str, Any]:
        """解析输入或输出的组成部分"""
        components = {}
        lines = text.strip().split('\n')
        current_tag = None
        current_content = []

        for line in lines:
            if line.endswith(':'):
                # 保存前一个标签的内容
                if current_tag and current_content:
                    components[current_tag] = '\n'.join(current_content).strip()
                    current_content = []
                current_tag = line[:-1].strip()  # 去掉冒号
            else:
                if current_tag:
                    current_content.append(line.strip())

        # 保存最后一个标签的内容
        if current_tag and current_content:
            components[current_tag] = '\n'.join(current_content).strip()

        return components

    def extract_output_tags_from_instruction(self, instruction: str) -> List[str]:
        """从instruction中提取输出标签"""
        # 分割输入和输出部分
        if '->' in instruction:
            output_part = instruction.split('->')[1]
        else:
            output_part = instruction

        # 分割多个输出标签
        tags = []
        for part in output_part.split('+'):
            tags.append(part.strip())

        return tags

    def parse_model_output(self, output_text: str, expected_tags: List[str]) -> Dict[str, Any]:
        """解析模型输出文本"""
        result = {
            "components": {},
            "missing_tags": [],
            "extra_tags": [],
            "is_resolved_correct": True,
            "parsing_errors": []
        }

        lines = output_text.strip().split('\n')
        current_tag = None
        current_content = []
        found_tags = set()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 检查是否为标签行（以冒号结尾或单独一行）
            if line.endswith(':') or (line in [tag.upper() for tag in expected_tags] or
                                      line in [tag.replace('.', '').upper() for tag in expected_tags]):

                # 保存前一个标签的内容
                if current_tag and current_content:
                    component_text = '\n'.join(current_content).strip()
                    if component_text:  # 确保有内容
                        result["components"][current_tag] = component_text
                        found_tags.add(current_tag)
                    else:
                        result["parsing_errors"].append(f"Tag '{current_tag}' has no content")
                        result["is_resolved_correct"] = False

                # 处理新标签
                if line.endswith(':'):
                    current_tag = line[:-1].strip()
                else:
                    current_tag = line

                # 标准化标签格式
                current_tag = self.standardize_tag(current_tag, expected_tags)
                current_content = []
                i += 1

                # 检查下一行是否立即是另一个标签（格式错误）
                if i < len(lines) and (lines[i].strip().endswith(':') or
                                       lines[i].strip() in [tag.upper() for tag in expected_tags] or
                                       lines[i].strip() in [tag.replace('.', '').upper() for tag in expected_tags]):
                    result["parsing_errors"].append(f"Tag '{current_tag}' has no content before next tag")
                    result["is_resolved_correct"] = False
                    # 不增加i，让下一轮处理这个标签
                else:
                    # 收集内容直到下一个标签
                    while i < len(lines) and not (
                            lines[i].strip().endswith(':') or
                            lines[i].strip() in [tag.upper() for tag in expected_tags] or
                            lines[i].strip() in [tag.replace('.', '').upper() for tag in expected_tags]
                    ):
                        current_content.append(lines[i].strip())
                        i += 1
            else:
                # 如果没有当前标签但遇到内容行，说明格式错误
                if not current_tag and line:
                    result["parsing_errors"].append(f"Content '{line}' without preceding tag")
                    result["is_resolved_correct"] = False
                i += 1

        # 处理最后一个标签
        if current_tag and current_content:
            component_text = '\n'.join(current_content).strip()
            if component_text:
                result["components"][current_tag] = component_text
                found_tags.add(current_tag)
            else:
                result["parsing_errors"].append(f"Tag '{current_tag}' has no content")
                result["is_resolved_correct"] = False

        # 检查缺失和多余的标签
        for expected_tag in expected_tags:
            standardized_expected = self.standardize_tag(expected_tag, expected_tags)
            if standardized_expected not in found_tags:
                result["missing_tags"].append(expected_tag)

        for found_tag in found_tags:
            # 反向查找原始标签
            original_tag = None
            for exp_tag in expected_tags:
                if self.standardize_tag(exp_tag, expected_tags) == found_tag:
                    original_tag = exp_tag
                    break
            if not original_tag:
                result["extra_tags"].append(found_tag)

        result["is_task_matched"] = len(result["missing_tags"]) == 0 and len(result["extra_tags"]) == 0

        return result

    def standardize_tag(self, tag: str, expected_tags: List[str]) -> str:
        """标准化标签格式"""
        # 移除可能的空格和特殊字符
        tag = tag.strip().upper()

        # 尝试匹配预期标签
        for expected_tag in expected_tags:
            expected_standardized = expected_tag.replace('.', '').upper()
            tag_standardized = tag.replace('.', '').upper()
            if expected_standardized == tag_standardized:
                return expected_tag

        # 如果没有匹配，返回原始标签的大写形式
        return tag

    def compare_by_inchikey(self, smiles1: str, smiles2: str) -> bool:
        """通过InchiKey比较分子是否相同"""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return False

            inchikey1 = Chem.MolToInchiKey(mol1)
            inchikey2 = Chem.MolToInchiKey(mol2)

            return inchikey1 == inchikey2
        except:
            return False

    def is_valid_smiles(self, smiles: str) -> bool:
        """检查SMILES是否有效"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def is_valid_reaction_smiles(self, rxn_smiles: str) -> bool:
        """检查反应SMILES是否有效"""
        try:
            if '>>' not in rxn_smiles:
                return False

            reactants, products = rxn_smiles.split('>>', 1)
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split('.')]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split('.')]

            return all(mol is not None for mol in reactant_mols + product_mols)
        except:
            return False

    def evaluate_component(self, component_type: str, predicted: str, ground_truth: str,
                           input_components: Dict[str, str] = None) -> Dict[str, Any]:
        """评估单个组件的正确性"""
        result = {
            "text": predicted,
            "is_correct": False
        }

        # 根据组件类型添加不同的评估指标
        if component_type in ["CLS", "MECH"]:
            result["is_correct"] = predicted.strip() == ground_truth.strip()

        elif any(x in component_type for x in ["RXTS", "PRDS", "RXN"]):
            # 初始化评估指标
            result.update({
                "is_correct_text": False,
                "is_correct_mols": False,
                "is_correct_format": False,
                "is_correct_manner": False,
                "is_valid": True
            })

            # 检查格式和类型一致性
            is_cano = "CANO" in component_type
            is_am = "AM" in component_type
            is_std = "STD" in component_type

            # 检查格式（STD vs AM）
            pred_has_mapping = re.search(r':\d+', predicted) is not None
            gt_has_mapping = re.search(r':\d+', ground_truth) is not None

            if is_am:
                result["is_correct_format"] = pred_has_mapping and gt_has_mapping
            elif is_std:
                result["is_correct_format"] = not pred_has_mapping and not gt_has_mapping
            else:
                result["is_correct_format"] = pred_has_mapping == gt_has_mapping

            if is_cano:
                result["is_correct_text"] = predicted.strip() == ground_truth.strip()
                if result["is_correct_text"]:
                    result["is_correct"] = True
                    result["is_correct_mols"] = True
                    result["is_correct_manner"] = True
                    result["is_valid"] = True
                    return result

            if is_am and "PRDS" in component_type and input_components:
                # 对于AM PRDS，需要比对atom mapping关系
                # 查找对应的反应物信息
                reactants_key = None
                for key in input_components.keys():
                    if "RXTS" in key:
                        reactants_key = key
                        break

                if reactants_key:
                    reactants_smiles = input_components[reactants_key]
                    # 比较atom mapping关系
                    mapping_correct = compare_atom_mapping_relationship(
                        predicted, ground_truth, reactants_smiles
                    )
                    result["is_correct_mols"] = mapping_correct
                else:
                    result["is_correct_mols"] = self.compare_by_inchikey(predicted, ground_truth)
                    # 验证分子有效性
                result["is_valid"] = self.is_valid_smiles(predicted)
                result["is_correct_manner"] = self.check_manner_consistency(predicted, "CANO" if is_cano else "ARBI")
            elif is_am and "RXN" in component_type:
                # 对于AM RXN，比较反应式的atom mapping
                mapping_correct = compare_reaction_atom_mapping(predicted, ground_truth)
                result["is_correct_mols"] = mapping_correct
                result["is_valid"] = self.is_valid_reaction_smiles(predicted)
                pred_reactants, pred_products = predicted.split('>>', 1)
                result["is_correct_manner"] = (self.check_manner_consistency(pred_reactants, "CANO") and
                                               self.check_manner_consistency(pred_products, "CANO"))
            else:
                # 其他情况可以使用InchiKey比对
                result["is_correct_mols"] = self.compare_by_inchikey(predicted, ground_truth)

            # 综合判断is_correct
            result["is_correct"] = (result["is_correct_mols"] and
                                    result["is_correct_format"] and
                                    result["is_correct_manner"])

        return result


    def check_manner_consistency(self, smiles: str, expected_manner: str) -> bool:
        """检查SMILES方式一致性（CANO vs ARBI）"""
        # 简化检查：CANO通常更标准化，ARBI可能有更多变体
        # 这里使用简单的启发式方法
        if expected_manner.upper() == "CANO":
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canonical_smiles = Chem.MolToSmiles(mol)
                    return smiles == canonical_smiles
            except:
                pass
        return False

    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个数据项"""
        processed_item = item.copy()

        # 获取任务信息
        instruction = item["instruction"]
        expected_output_tags = self.extract_output_tags_from_instruction(instruction)

        # 解析输入组件
        input_components = self.parse_input_output_components(item["input"], "input")

        # 解析标签组件
        label_components = self.parse_input_output_components(item["label"]["text"], "label")
        processed_item["label"]["components"] = label_components
        processed_item["label"]["output_tags"] = expected_output_tags

        # 处理模型输出
        processed_outputs = []
        for output in item["output"]:
            output_text = output["text"]
            parsed_output = self.parse_model_output(output_text, expected_output_tags)

            processed_output = {
                "text": output_text,
                "length": output["length"],
                "sequence_score": output["sequence_score"]
            }

            # 评估每个组件
            all_components_correct = True
            for tag in expected_output_tags:
                if tag in parsed_output["components"] and tag in label_components:
                    evaluation = self.evaluate_component(
                        tag,
                        parsed_output["components"][tag],
                        label_components[tag],
                        input_components
                    )
                    processed_output[tag.replace(".", "_").lower()] = evaluation

                    if not evaluation.get("is_correct", False):
                        all_components_correct = False
                else:
                    # 如果标签缺失，则标记为不正确
                    processed_output[tag] = {
                        "text": parsed_output["components"].get(tag, ""),
                        "is_correct": False
                    }
                    all_components_correct = False

            # 添加解析结果
            processed_output.update({
                "missing_tags": parsed_output["missing_tags"],
                "extra_tags": parsed_output["extra_tags"],
                "is_correct_totally": all_components_correct and parsed_output["is_resolved_correct"],
                "is_resolved_correct": parsed_output["is_resolved_correct"],
                "is_task_matched": parsed_output["is_task_matched"]
            })

            processed_outputs.append(processed_output)

        processed_item["output"] = processed_outputs
        return processed_item

    def clean_filename(self, filename: str) -> str:
        """清理文件名，去除_para01等后缀"""
        return re.sub(r'_para\d+', '', filename)

    def process_file(self, file_path: Path, output_path: Path):
        """处理单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            processed_data = []
            for item in data:
                processed_item = self.process_single_item(item)
                processed_data.append(processed_item)

            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            logger.info(f"成功处理文件: {file_path} -> {output_path}")

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")

    def process_all_files(self):
        """处理所有文件"""
        all_files = []

        # 收集所有需要处理的文件
        for folder in self.process_folders:
            folder_path = self.base_dir / folder
            if folder_path.exists():
                for json_file in folder_path.rglob("*.json"):
                    all_files.append(json_file)
            else:
                logger.warning(f"文件夹不存在: {folder_path}")

        logger.info(f"找到 {len(all_files)} 个需要处理的文件")

        # 处理文件
        for file_path in tqdm(all_files, desc="处理文件"):
            # 计算相对路径和输出路径
            relative_path = file_path.relative_to(self.base_dir)
            cleaned_filename = self.clean_filename(relative_path.name)
            output_file_path = self.output_dir / relative_path.parent / cleaned_filename

            self.process_file(file_path, output_file_path)


def main():
    parser = argparse.ArgumentParser(description="处理化学机械预测数据")
    parser.add_argument("--base_dir", default=BASE_DIR, help="输入数据目录")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="输出数据目录")
    parser.add_argument("--process_folders", nargs="+", default=PROCESS_FOLDERS,
                        help="需要处理的文件夹列表")

    args = parser.parse_args()

    logger.info("开始处理数据...")
    logger.info(f"输入目录: {args.base_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"处理文件夹: {args.process_folders}")

    processor = DataProcessor(args.base_dir, args.output_dir, args.process_folders)
    processor.process_all_files()

    logger.info("数据处理完成!")


if __name__ == "__main__":
    main()