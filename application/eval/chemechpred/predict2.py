# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: predict2.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/27 10:07
# https://chat.deepseek.com/a/chat/s/2230b7ad-64ef-449d-a448-338ee9ceef94

# -*- coding: utf-8 -*-
import gc
import os
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
import time
import GPUtil

import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset
from llamafactory.chat import ChatModel

# ============================ 配置常量 ============================
GPU_MEMORY_THRESHOLD = 8000  # MB，GPU显存阈值
LOG_DIR = "./logs/chemechpred"

# Vaguely Defined 任务标识符
VAGUELY_DEFINED_TASKS = {
    # RXN_TO_MECH
    "RXN->CLS", "RXN->MECH", "RXN->CLS+MECH",
    # RXN_TO_RXN
    "ORI.RXN->UPD.RXN", "STD.RXN->AM.RXN", "AM.RXN->STD.RXN",
    "ARBI.RXN->CANO.RXN", "CANO.RXN->ARBI.RXN",
    # RXTS_TO_RXTS
    "STD.RXTS->AM.RXTS", "AM.RXTS->STD.RXTS",
    "ARBI.RXTS->CANO.RXTS", "CANO.RXTS->ARBI.RXTS",
    # PRDS_TO_PRDS
    "AM.PRDS->STD.PRDS", "ARBI.PRDS->CANO.PRDS", "CANO.PRDS->ARBI.PRDS",
    # RXTS_TO_PRDS
    "RXTS->PRDS",
    # PRDS_TO_RXTS
    "PRDS->RXTS",
    # STYLE_TO_STYLE (全部)
    "CANO->ARBI", "ARBI->CANO", "STD->AM", "AM->STD"
}


# ============================ 工具函数 ============================
def wait_for_gpu_memory(threshold_mb: int = GPU_MEMORY_THRESHOLD):
    """等待GPU显存达到阈值"""
    while True:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No GPU found, proceeding with CPU...")
            break

        available_memory = min([gpu.memoryFree for gpu in gpus])
        if available_memory >= threshold_mb:
            print(f"GPU memory available: {available_memory}MB")
            break
        else:
            print(f"Waiting for GPU memory... (available: {available_memory}MB, required: {threshold_mb}MB)")
            time.sleep(60)


def parse_output_text(text: str, expected_output_tags: List[str]) -> Dict[str, Any]:
    """
    解析模型输出的文本，提取各个信息模块

    Returns:
        Dict containing:
            - parsed_results: 解析出的各个信息模块 {tag: text}
            - is_resolved_correct: 解析过程是否正确
            - is_task_matched: 是否匹配任务要求
            - missing_tags: 缺失的标签
            - extra_tags: 多余的标签
    """
    # 使用正则表达式匹配信息标识符（以冒号结尾的单词）
    pattern = r'([A-Za-z_\.]+:)\s*\n'
    matches = list(re.finditer(pattern, text))

    parsed_results = {}
    sections = []

    # 提取各个信息段
    for i, match in enumerate(matches):
        tag = match.group(1).rstrip(':')
        start_pos = match.end()

        # 查找下一个标签的位置或文本结束
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        content = text[start_pos:end_pos].strip()
        sections.append((tag, content))

    # 如果没有找到标签，尝试按行分割
    if not sections:
        lines = text.strip().split('\n')
        current_tag = None
        current_content = []

        for line in lines:
            if line.endswith(':'):
                if current_tag and current_content:
                    sections.append((current_tag, '\n'.join(current_content)))
                current_tag = line.rstrip(':')
                current_content = []
            else:
                current_content.append(line)

        if current_tag and current_content:
            sections.append((current_tag, '\n'.join(current_content)))

    # 转换为字典
    for tag, content in sections:
        parsed_results[tag.lower().replace('.', '_').replace('->', '_to_')] = content

    # 检查解析正确性
    is_resolved_correct = len(sections) > 0

    # 检查任务匹配度
    expected_lower = [tag.lower().replace('.', '_').replace('->', '_to_') for tag in expected_output_tags]
    found_tags = list(parsed_results.keys())

    missing_tags = [tag for tag in expected_lower if tag not in found_tags]
    extra_tags = [tag for tag in found_tags if tag not in expected_lower]
    is_task_matched = len(missing_tags) == 0

    return {
        "parsed_results": parsed_results,
        "is_resolved_correct": is_resolved_correct,
        "is_task_matched": is_task_matched,
        "missing_tags": missing_tags,
        "extra_tags": extra_tags
    }


def compare_molecules_smiles(smiles1: str, smiles2: str) -> bool:
    """比较两个SMILES字符串是否代表相同的分子"""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return False

        return Chem.MolToInchiKey(mol1) == Chem.MolToInchiKey(mol2)
    except:
        return False


def is_valid_smiles(smiles: str) -> bool:
    """检查SMILES字符串是否有效"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def is_valid_reaction_smiles(rxn_smiles: str) -> bool:
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


def check_format_consistency(smiles: str, expected_format: str) -> bool:
    """检查SMILES格式一致性（STD vs AM）"""
    has_mapping = re.search(r':\d+', smiles) is not None
    if expected_format.upper() == "AM":
        return has_mapping
    else:  # STD
        return not has_mapping


def check_manner_consistency(smiles: str, expected_manner: str) -> bool:
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
    return False  # 对于ARBI，不进行严格检查


def evaluate_output_component(predicted: str, reference: str, component_type: str,
                              expected_format: Optional[str] = None,
                              expected_manner: Optional[str] = None) -> Dict[str, Any]:
    """
    评估单个输出组件

    Args:
        predicted: 预测的文本
        reference: 参考文本
        component_type: 组件类型 ('cls', 'mech', 'rxn', 'rxts', 'prds')
        expected_format: 期望的格式 ('STD', 'AM')
        expected_manner: 期望的方式 ('CANO', 'ARBI')
    """
    result = {
        "text": predicted,
        "is_correct": False,
        "is_correct_text": False,
        "is_correct_mols": False,
        "is_correct_format": False,
        "is_correct_manner": False,
        "is_valid": False
    }

    # 文本匹配
    result["is_correct_text"] = (predicted.strip() == reference.strip())

    if component_type in ['cls', 'mech']:
        # CLS和MECH只检查文本匹配
        result["is_correct"] = result["is_correct_text"]
        result["is_valid"] = True  # 文本总是有效的

    elif component_type in ['rxts', 'prds']:
        # 检查分子有效性
        result["is_valid"] = is_valid_smiles(predicted)

        # 检查分子一致性
        if result["is_valid"] and is_valid_smiles(reference):
            result["is_correct_mols"] = compare_molecules_smiles(predicted, reference)

        # 检查格式和方式
        if expected_format:
            result["is_correct_format"] = check_format_consistency(predicted, expected_format)
        if expected_manner:
            result["is_correct_manner"] = check_manner_consistency(predicted, expected_manner)

        # 综合判断
        if 'cano' in component_type.lower():
            result["is_correct"] = result["is_correct_text"]
        else:
            result["is_correct"] = (result["is_correct_mols"] and
                                    result["is_correct_format"] and
                                    result["is_correct_manner"])

    elif component_type == 'rxn':
        # 反应检查
        result["is_valid"] = is_valid_reaction_smiles(predicted)

        if result["is_valid"] and is_valid_reaction_smiles(reference):
            # 分割反应物和产物
            pred_reactants, pred_products = predicted.split('>>', 1)
            ref_reactants, ref_products = reference.split('>>', 1)

            # 检查分子一致性
            pred_reactant_mols = [smi.strip() for smi in pred_reactants.split('.')]
            ref_reactant_mols = [smi.strip() for smi in ref_reactants.split('.')]
            pred_product_mols = [smi.strip() for smi in pred_products.split('.')]
            ref_product_mols = [smi.strip() for smi in ref_products.split('.')]

            # 简化的分子集合比较
            reactant_match = (len(pred_reactant_mols) == len(ref_reactant_mols) and
                              all(any(compare_molecules_smiles(p, r) for r in ref_reactant_mols)
                                  for p in pred_reactant_mols))
            product_match = (len(pred_product_mols) == len(ref_product_mols) and
                             all(any(compare_molecules_smiles(p, r) for r in ref_product_mols)
                                 for p in pred_product_mols))

            result["is_correct_mols"] = reactant_match and product_match

            # 检查格式和方式
            if expected_format:
                result["is_correct_format"] = (check_format_consistency(pred_reactants, expected_format) and
                                               check_format_consistency(pred_products, expected_format))
            if expected_manner:
                result["is_correct_manner"] = (check_manner_consistency(pred_reactants, expected_manner) and
                                               check_manner_consistency(pred_products, expected_manner))

        # 综合判断
        if 'cano' in component_type.lower():
            result["is_correct"] = result["is_correct_text"]
        else:
            result["is_correct"] = (result["is_correct_mols"] and
                                    result["is_correct_format"] and
                                    result["is_correct_manner"])

    return result


def extract_metadata_from_tag(tag: str) -> Tuple[Optional[str], Optional[str]]:
    """从标签中提取格式和方式信息"""
    format_info = None
    manner_info = None

    tag_upper = tag.upper()
    if 'STD' in tag_upper:
        format_info = 'STD'
    elif 'AM' in tag_upper:
        format_info = 'AM'

    if 'CANO' in tag_upper:
        manner_info = 'CANO'
    elif 'ARBI' in tag_upper:
        manner_info = 'ARBI'

    return format_info, manner_info


# ============================ 数据集类 ============================
class DynamicBatchDataset(Dataset):
    """支持动态批量处理的数据集"""

    def __init__(self, data_path: str, tokenizer):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        self.samples = []
        for idx, item in enumerate(raw_data):
            prompt = f"{item['instruction']}\n{item['input']}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            self.samples.append({
                "id": item["id"],
                "original_idx": idx,
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
                "length": len(prompt_ids)
            })

        # 按长度排序以便动态批量处理
        self.samples.sort(key=lambda x: x["length"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def create_batches(self, batch_limit: int, batch_token_size: int, minmax_gap: int):
        """根据约束条件创建动态批量"""
        batches = []
        current_batch = []
        current_batch_size = 0
        current_max_len = 0

        for sample in self.samples:
            sample_len = sample["length"]

            # 检查是否可加入当前batch
            if (len(current_batch) < batch_limit and
                    current_batch_size + sample_len <= batch_token_size and
                    (not current_batch or sample_len - current_batch[0]["length"] <= minmax_gap)):

                current_batch.append(sample)
                current_batch_size += sample_len
                current_max_len = max(current_max_len, sample_len)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sample]
                current_batch_size = sample_len
                current_max_len = sample_len

        if current_batch:
            batches.append(current_batch)

        return batches


# ============================ 主预测类 ============================
class ChemMechPredictor:
    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.setup_logging()

    def setup_paths(self):
        """设置文件路径"""
        # 解析task_id获取group信息
        task_id = self.args.task_id.upper()
        self.group = self.args.group
        self.task_id = task_id.lower()

        if "arbi" in task_id.lower():
            augm = "augm_x1"
        else:
            augm = "augm_x1"

        # 设置数据路径
        self.data_dir = Path(self.args.data_base_dir) / self.args.subset / self.group.lower() / augm
        self.data_file = self.data_dir / f"{self.task_id}.json"

        if not self.data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")

        # 设置输出路径
        self.output_dir = Path(self.args.output_base_dir) / self.args.subset / self.group.lower() /self.task_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.args.model_name}.json"

    def setup_logging(self):
        """设置日志"""
        log_dir = Path(LOG_DIR) / self.group.lower()
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.args.model_name}_{timestamp}.log"

        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def parse_reference_output(self, output_text: str, instruction: str) -> Dict[str, Any]:
        """解析参考输出"""
        # 从instruction中提取输出标签
        output_part = instruction.split('->')[1]
        output_tags = [tag.strip() for tag in output_part.split('+')]

        parsed = parse_output_text(output_text, output_tags)
        reference_components = {}

        for tag in output_tags:
            tag_lower = tag.lower().replace('.', '_')
            if tag_lower in parsed["parsed_results"]:
                reference_components[tag_lower] = parsed["parsed_results"][tag_lower]
            else:
                reference_components[tag_lower] = ""

        return {
            "text": output_text,
            "components": reference_components,
            "output_tags": output_tags
        }

    def evaluate_prediction(self, predicted_text: str, reference_data: Dict,
                            instruction: str) -> Dict[str, Any]:
        """评估单个预测结果"""
        # 确定任务类型
        task_tag = instruction
        is_vaguely_defined = task_tag in VAGUELY_DEFINED_TASKS

        # 解析预测输出
        output_tags = reference_data["output_tags"]
        parsed_pred = parse_output_text(predicted_text, output_tags)

        evaluation_results = {
            "text": predicted_text,
            "is_resolved_correct": parsed_pred["is_resolved_correct"],
            "is_task_matched": parsed_pred["is_task_matched"],
            "missing_tags": parsed_pred["missing_tags"],
            "extra_tags": parsed_pred["extra_tags"]
        }

        # 评估各个组件
        component_results = {}
        all_correct = True

        for tag in output_tags:
            tag_lower = tag.lower().replace('.', '_')
            predicted_value = parsed_pred["parsed_results"].get(tag_lower, "")
            reference_value = reference_data["components"].get(tag_lower, "")

            # 确定组件类型和元数据
            component_type = self.determine_component_type(tag)
            expected_format, expected_manner = extract_metadata_from_tag(tag)

            # 对于模糊定义的任务，需要从ori_tag中获取元数据
            if is_vaguely_defined and hasattr(self, 'current_sample') and 'ori_tag' in self.current_sample:
                ori_tag = self.current_sample.get('ori_tag', '')
                ori_format, ori_manner = extract_metadata_from_tag(ori_tag)
                if ori_format:
                    expected_format = ori_format
                if ori_manner:
                    expected_manner = ori_manner

            # 评估组件
            component_eval = evaluate_output_component(
                predicted_value, reference_value, component_type,
                expected_format, expected_manner
            )

            component_results[tag_lower] = component_eval

            if not component_eval["is_correct"]:
                all_correct = False

        evaluation_results.update(component_results)
        evaluation_results["is_correct_totally"] = all_correct

        return evaluation_results

    def determine_component_type(self, tag: str) -> str:
        """根据标签确定组件类型"""
        tag_lower = tag.lower()
        if 'cls' in tag_lower:
            return 'cls'
        elif 'mech' in tag_lower:
            return 'mech'
        elif 'rxn' in tag_lower and 'rxts' not in tag_lower and 'prds' not in tag_lower:
            return 'rxn'
        elif 'rxts' in tag_lower:
            return 'rxts'
        elif 'prds' in tag_lower:
            return 'prds'
        else:
            return 'unknown'

    def process_responses(self, responses: List, reference_output: str, instruction: str) -> List[Dict]:
        """处理模型响应"""
        reference_data = self.parse_reference_output(reference_output, instruction)
        processed = []

        for resp in responses:
            text = resp.response_text
            evaluation = self.evaluate_prediction(text, reference_data, instruction)

            item = {
                "text": text,
                "length": resp.response_length,
                "sequence_score": resp.sequence_score,
                **evaluation
            }
            processed.append(item)

        # 按分数降序排列
        processed.sort(key=lambda x: x["sequence_score"], reverse=True)
        return processed

    def run_prediction(self):
        """运行预测"""
        self.logger.info("开始化学机制预测任务")
        self.logger.info(f"任务ID: {self.task_id}")
        self.logger.info(f"模型: {self.args.model_name}")
        self.logger.info(f"数据文件: {self.data_file}")

        # 等待GPU内存
        if self.args.wait_for_gpu:
            wait_for_gpu_memory(self.args.gpu_threshold)

        # 加载模型
        infer_args = {
            "model_name_or_path": str(Path(self.args.model_path) / self.args.section / self.args.model_name),
            "finetuning_type": self.args.finetuning_type,
            "template": self.args.template,
            "num_beams": self.args.num_beams,
            "do_sample": self.args.do_sample,
            "max_new_tokens": self.args.max_new_tokens,
        }

        chat_model = ChatModel(infer_args)
        tokenizer = chat_model.engine.tokenizer

        # 加载数据集
        dataset = DynamicBatchDataset(str(self.data_file), tokenizer)
        batches = dataset.create_batches(self.args.batch_limit, self.args.batch_token_size, self.args.minmax_gap)

        results = [None] * len(dataset)

        # 批量处理
        for batch in tqdm(batches, desc="Processing batches"):
            messages = [[{
                "role": "user",
                "content": f"{sample['instruction']}\n{sample['input']}"
            }] for sample in batch]

            batch_responses = chat_model.batch_llm_predict(
                messages,
                num_return_sequences=self.args.num_return_sequences,
                output_scores=self.args.output_scores,
                return_dict_in_generate=self.args.return_dict_in_generate
            )

            for sample, responses in zip(batch, batch_responses):
                self.current_sample = sample  # 用于模糊定义任务
                processed = self.process_responses(responses, sample["output"], sample["instruction"])

                # 解析参考输出用于结果存储
                reference_data = self.parse_reference_output(sample["output"], sample["instruction"])

                results[sample["original_idx"]] = {
                    "id": sample["id"],
                    "instruction": sample["instruction"],
                    "input": sample["input"],
                    "prompt_length": sample["length"],
                    "label": reference_data,
                    "output": processed
                }

        # 保存结果
        results = [r for r in results if r is not None]
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"预测完成，结果保存至: {self.output_file}")

        # 清理资源
        del chat_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="化学机制预测脚本")

    # 必需参数
    parser.add_argument("--group", type=str, default="",
                        help="group")
    parser.add_argument("--task_id", type=str, default="UPDCANOAMPRDS_TO_UPDCANOSTDPRDS",
                        help="任务ID，如ORICANOAMRXN_TO_UPDCANOAMRXN")
    # 模型参数
    parser.add_argument("--model_path", type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory/chemechpred/")
    parser.add_argument("--section", type=str,
                        default="prds_to_prds")
    parser.add_argument("--model_name", type=str, default="updcanoamprds_to_updcanostdprds",
                        help="模型名称")

    # 数据路径参数
    parser.add_argument("--data_base_dir", type=str,
                        default="DataSets/Chemistry/ChemicalMechanism/via_random/test/")
    parser.add_argument("--subset", type=str, default="_random313")
    parser.add_argument("--output_base_dir", type=str,
                        default="results/chemechpred/prediction")

    # 推理参数
    parser.add_argument("--finetuning_type", type=str, default="full")
    parser.add_argument("--template", type=str, default="qwen")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--output_scores", action="store_true", default=True)
    parser.add_argument("--return_dict_in_generate", action="store_true", default=True)

    # 动态批处理参数
    parser.add_argument("--batch_limit", type=int, default=2,
                        help="Maximum number of sequences in a batch")
    parser.add_argument("--batch_token_size", type=int, default=2000,
                        help="Maximum total token size per batch")
    parser.add_argument("--minmax_gap", type=int, default=40,
                        help="Maximum allowed length difference within a batch")

    # GPU参数
    parser.add_argument("--wait_for_gpu", action="store_true", default=True,
                        help="是否等待GPU内存达到阈值")
    parser.add_argument("--gpu_threshold", type=int, default=GPU_MEMORY_THRESHOLD,
                        help="GPU内存阈值(MB)")

    args = parser.parse_args()

    # 运行预测
    predictor = ChemMechPredictor(args)
    predictor.run_prediction()


if __name__ == "__main__":
    main()