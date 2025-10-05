# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: predict.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/24 14:49
# https://chat.deepseek.com/a/chat/s/177595a4-7f84-4f2c-a0f8-6df75af2a02d
# -*- coding: utf-8 -*-
import gc
import os
import json
import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import Dataset
from llamafactory.chat import ChatModel


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


class TaskConfig:
    """任务配置类"""

    # 任务标识符映射
    TASK_MAPPING = {
        "UPDCANOAMRXTS_TO_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS"],
            "output_fields": ["UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS->UPD.CANO.AM.PRDS"
        },
        "UPDCANOAMRXTS_TO_MECH_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS"],
            "output_fields": ["MECH", "UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS->MECH+UPD.CANO.AM.PRDS"
        },
        "UPDCANOAMRXTS_TO_CLS_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS"],
            "output_fields": ["CLS", "UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS->CLS+UPD.CANO.AM.PRDS"
        },
        "UPDCANOAMRXTS_TO_CLS_MECH_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS"],
            "output_fields": ["CLS", "MECH", "UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS->CLS+MECH+UPD.CANO.AM.PRDS"
        },
        "UPDCANOAMRXTS_MECH_TO_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS", "MECH"],
            "output_fields": ["UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS+MECH->UPD.CANO.AM.PRDS"
        },
        "UPDCANOAMRXTS_CLS_TO_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS", "CLS"],
            "output_fields": ["UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS+CLS->UPD.CANO.AM.PRDS"
        },
        "UPDCANOAMRXTS_MECH_CLS_TO_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS", "MECH", "CLS"],
            "output_fields": ["UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS+MECH+CLS->UPD.CANO.AM.PRDS"
        },
        "UPDCANOAMRXTS_CLS_TO_MECH_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS", "CLS"],
            "output_fields": ["MECH", "UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS+CLS->MECH+UPD.CANO.AM.PRDS"
        },
        "UPDCANOAMRXTS_MECH_TO_CLS_UPDCANOAMPRDS": {
            "input_fields": ["UPDCANOAMRXTS", "MECH"],
            "output_fields": ["CLS", "UPDCANOAMPRDS"],
            "instruction_template": "UPD.CANO.AM.RXTS+MECH->CLS+UPD.CANO.AM.PRDS"
        }
    }

    # 字段显示名称映射
    FIELD_DISPLAY_NAMES = {
        "UPDCANOAMRXTS": "UPD.CANO.AM.RXTS",
        "UPDCANOAMPRDS": "UPD.CANO.AM.PRDS",
        "MECH": "MECH",
        "CLS": "CLS"
    }

    INFOID_MTRFRG_MAPPING = {
        "UPDCANOAMRXTS": "upd_cano_am_rxts",
        "UPDCANOAMPRDS": "upd_cano_am_prds",
        "MECH": "mech",
        "CLS": "cls"
    }

    @classmethod
    def get_task_config(cls, task_id: str) -> Dict[str, Any]:
        """获取任务配置"""
        if task_id not in cls.TASK_MAPPING:
            raise ValueError(f"未知的任务ID: {task_id}")
        return cls.TASK_MAPPING[task_id]

    @classmethod
    def get_field_display_name(cls, field_id: str) -> str:
        """获取字段显示名称"""
        return cls.FIELD_DISPLAY_NAMES.get(field_id, field_id)


class MolecularEvaluator:
    """分子评估器"""

    @staticmethod
    def is_valid_smiles(smiles: str) -> bool:
        """检查SMILES是否有效"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    @staticmethod
    def is_valid_atom_mapped_smiles(smiles: str) -> bool:
        """检查是否为有效的原子映射SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # 检查是否有原子映射编号
            pattern = r'\[\w+?:\d+\]'
            matches = re.findall(pattern, smiles)
            return len(matches) > 0
        except:
            return False

    @staticmethod
    def are_identical_molecules(smiles1: str, smiles2: str) -> bool:
        """检查两个SMILES是否代表相同的分子"""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return False

            # 标准化SMILES进行比较
            canon_smiles1 = Chem.MolToSmiles(mol1, canonical=True)
            canon_smiles2 = Chem.MolToSmiles(mol2, canonical=True)

            return canon_smiles1 == canon_smiles2
        except:
            return False

    @staticmethod
    def canonicalize_smiles(smiles: str) -> str:
        """标准化SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return smiles


class ResponseParser:
    """响应解析器"""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.task_config = TaskConfig.get_task_config(task_id)
        self.output_fields = self.task_config["output_fields"]

    def parse_model_output(self, text: str) -> Dict[str, Any]:
        """解析模型输出"""
        result = {"text": text, "parsed_fields": {}, "parsing_errors": []}

        # 按行分割
        lines = text.strip().split('\n')
        current_field = None
        current_content = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 检查是否为字段标识行
            field_match = self._identify_field(line)
            if field_match:
                # 保存前一个字段的内容
                if current_field and current_content:
                    result["parsed_fields"][current_field] = '\n'.join(current_content).strip()
                    current_content = []

                current_field = field_match

                # 检查字段标识符后是否有内容（语法错误：字段标识符和内容在同一行）
                remaining_content = line.replace(TaskConfig.get_field_display_name(field_match), "").strip()
                if remaining_content and not remaining_content.endswith(':'):
                    result["parsing_errors"].append(f"字段'{field_match}'标识符后直接跟内容，缺少分隔符")
                    current_content.append(remaining_content)
            else:
                if current_field:
                    current_content.append(line)
                else:
                    # 没有当前字段但遇到内容行，说明有解析错误
                    result["parsing_errors"].append(f"第{i + 1}行内容没有对应的字段标识符: '{line}'")

        # 处理最后一个字段
        if current_field and current_content:
            result["parsed_fields"][current_field] = '\n'.join(current_content).strip()

        # 检查是否缺少必需的输出字段
        missing_fields = set(self.output_fields) - set(result["parsed_fields"].keys())
        for field in missing_fields:
            result["parsing_errors"].append(f"缺少必需的输出字段: {field}")

        return result

    def _identify_field(self, line: str) -> Optional[str]:
        """识别字段"""
        line_upper = line.upper()

        for field in self.output_fields:
            display_name = TaskConfig.get_field_display_name(field)
            # 检查字段标识符（带或不带冒号）
            if display_name.upper() in line_upper or f"{display_name.upper()}:" in line_upper:
                return field

        return None


class ChemMechPredictionEvaluator:
    """化学机制预测评估器"""

    def __init__(self, args):
        self.args = args
        self.molecular_evaluator = MolecularEvaluator()
        self.response_parser = ResponseParser(args.task_id)

    def evaluate_response(self, response_text: str, label_text: str) -> Dict[str, Any]:
        """评估单个响应"""
        # 解析模型输出
        parsed_output = self.response_parser.parse_model_output(response_text)

        # 解析标签
        parsed_label = self._parse_label(label_text)

        # 初始化评估结果
        eval_result = {
            "text": response_text,
            "parsed_fields": parsed_output["parsed_fields"],
            "parsing_errors": parsed_output["parsing_errors"],
            "is_task_matched": self._check_task_matching(parsed_output["parsed_fields"]),
            "is_resolved_correct": len(parsed_output["parsing_errors"]) == 0  # 没有解析错误即为正确解析
        }

        # 对每个输出字段进行评估
        total_correct = True
        for field in self.response_parser.output_fields:
            field_display = TaskConfig.get_field_display_name(field)
            pred_value = parsed_output["parsed_fields"].get(field, "")
            true_value = parsed_label.get(field, "")

            # 设置字段特定的评估指标
            if field == "UPDCANOAMPRDS":
                eval_result.update(self._evaluate_upd_cano_am_prds(pred_value, true_value))
            else:
                eval_result.update(self._evaluate_general_field(field, pred_value, true_value))
            # 更新总体正确性
            is_correct_key = f"is_correct_{TaskConfig.INFOID_MTRFRG_MAPPING[field]}"
            if field in self.response_parser.output_fields and not eval_result[is_correct_key]:
                total_correct = False

        eval_result["is_correct_totally"] = total_correct

        return eval_result

    def _parse_label(self, label_text: str) -> Dict[str, str]:
        """解析标签文本"""
        parsed = {}
        lines = label_text.strip().split('\n')
        current_field = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            field_match = self._identify_label_field(line)
            if field_match:
                # 保存前一个字段的内容
                if current_field and current_content:
                    parsed[current_field] = '\n'.join(current_content).strip()
                    current_content = []

                current_field = field_match

                # 处理字段标识符后的内容
                remaining_content = line.replace(TaskConfig.get_field_display_name(field_match), "").strip()
                if remaining_content:
                    # 移除可能的冒号
                    if remaining_content.startswith(':'):
                        remaining_content = remaining_content[1:].strip()
                    if remaining_content:
                        current_content.append(remaining_content)
            else:
                if current_field:
                    current_content.append(line)

        # 处理最后一个字段
        if current_field and current_content:
            parsed[current_field] = '\n'.join(current_content).strip()

        return parsed

    def _identify_label_field(self, line: str) -> Optional[str]:
        """识别标签中的字段"""
        line_upper = line.upper()

        for field in self.response_parser.output_fields:
            display_name = TaskConfig.get_field_display_name(field)
            # 检查字段标识符（带或不带冒号）
            if display_name.upper() in line_upper or f"{display_name.upper()}:" in line_upper:
                return field

        return None

    def _check_task_matching(self, parsed_fields: Dict[str, str]) -> bool:
        """检查任务匹配"""
        # 检查是否包含所有必需的输出字段
        required_fields = set(self.response_parser.output_fields)
        actual_fields = set(parsed_fields.keys())

        return required_fields.issubset(actual_fields)

    def _evaluate_upd_cano_am_prds(self, pred_value: str, true_value: str) -> Dict[str, Any]:
        """评估UPDCANOAMPRDS字段"""
        result = {}

        # 有效性检查
        is_valid = self.molecular_evaluator.is_valid_smiles(pred_value)
        # is_valid_am = self.molecular_evaluator.is_valid_atom_mapped_smiles(pred_value)

        # 正确性检查
        is_correct = False
        is_correct_canonical = False
        if is_valid and true_value:
            # 处理多个分子用"."连接的情况
            pred_molecules = pred_value.split('.')
            true_molecules = true_value.split('.')
            is_correct_canonical = sorted(pred_molecules) == sorted(true_molecules)
            if not is_correct_canonical:
                # 对每个分子进行标准化比较
                pred_canon = [self.molecular_evaluator.canonicalize_smiles(mol) for mol in pred_molecules]
                true_canon = [self.molecular_evaluator.canonicalize_smiles(mol) for mol in true_molecules]
                # 排序后比较（忽略顺序）
                is_correct = sorted(pred_canon) == sorted(true_canon)
            else:
                is_correct = True

        result.update({
            "upd_cano_am_prds": pred_value,
            "is_correct_upd_cano_am_prds": is_correct_canonical ,
            "is_valid_upd_cano_am_prds": is_valid,
            "is_correct_upd_am_prds": is_correct
        })

        return result

    def _evaluate_general_field(self, field: str, pred_value: str, true_value: str) -> Dict[str, Any]:
        """评估一般字段（CLS, MECH等）"""
        field_lower = field.lower()

        # 简单字符串比较（可扩展为更复杂的比较逻辑）
        is_correct = pred_value.strip().upper() == true_value.strip().upper()

        return {
            field_lower: pred_value,
            f"is_correct_{field_lower}": is_correct
        }


def wait_for_gpu_memory(threshold_gb: float = 10.0, check_interval: int = 30):
    """等待GPU内存可用"""
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过GPU内存检查")
        return

    while True:
        torch.cuda.empty_cache()
        # free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_memory_gb = free_memory / (1024 ** 3)

        if free_memory_gb >= threshold_gb:
            print(f"GPU内存可用: {free_memory_gb:.2f} GB")
            break

        print(f"GPU内存不足: {free_memory_gb:.2f} GB < {threshold_gb} GB，等待{check_interval}秒...")
        time.sleep(check_interval)


def setup_logging(model_name: str, log_dir: str = "./logs/chemechpred"):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def main(args):
    """主函数"""
    # 设置日志
    logger = setup_logging(args.model_name, args.log_dir)
    logger.info(f"开始评估任务: {args.task_id}")

    # 检查GPU内存
    if args.gpu_threshold > 0:
        wait_for_gpu_memory(args.gpu_threshold)

    # 构建数据路径
    data_file = f"{args.task_id.lower()}.json"
    data_path = os.path.join(args.data_dir, data_file)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 构建输出路径
    output_dir = os.path.join(args.output_dir, args.task_id.lower())
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.model_name}.json")

    # 初始化模型
    infer_args = {
        "model_name_or_path": str(Path(args.model_path) / args.model_name),
        "finetuning_type": args.finetuning_type,
        "template": args.template,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
    }

    chat_model = ChatModel(infer_args)
    tokenizer = chat_model.engine.tokenizer

    # 加载数据集
    dataset = DynamicBatchDataset(data_path, tokenizer)
    batches = dataset.create_batches(args.batch_limit, args.batch_token_size, args.minmax_gap)

    # 初始化评估器
    evaluator = ChemMechPredictionEvaluator(args)

    results = [None] * len(dataset)

    # 处理批次
    for batch in tqdm(batches, desc="Processing batches"):
        messages = [[{
            "role": "user",
            "content": f"{sample['instruction']}\n{sample['input']}"
        }] for sample in batch]

        batch_responses = chat_model.batch_llm_predict(
            messages,
            num_return_sequences=args.num_return_sequences,
            output_scores=args.output_scores,
            return_dict_in_generate=args.return_dict_in_generate
        )

        for sample, responses in zip(batch, batch_responses):
            processed_responses = []
            for resp in responses:
                # 评估每个响应
                eval_result = evaluator.evaluate_response(resp.response_text, sample["output"])
                eval_result.update({
                    "length": resp.response_length,
                    "sequence_score": resp.sequence_score
                })
                processed_responses.append(eval_result)

            # 按分数排序
            processed_responses.sort(key=lambda x: x["sequence_score"], reverse=True)

            # 准备标签解析
            label_parser = ResponseParser(args.task_id)
            parsed_label = label_parser.parse_model_output(sample["output"])
            label_dict = {"text": sample["output"]}
            label_dict.update(parsed_label["parsed_fields"])

            results[sample["original_idx"]] = {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "input": sample["input"],
                "prompt_length": sample["length"],
                "label": label_dict,
                "output": processed_responses
            }

    # 保存结果
    results = [r for r in results if r is not None]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"评估完成，结果保存至: {output_file}")

    # 清理资源
    del chat_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="化学机制预测评估脚本")

    # 必需参数
    parser.add_argument("--task_id", type=str, default="UPDCANOAMRXTS_CLS_TO_MECH_UPDCANOAMPRDS",
                        choices=list(TaskConfig.TASK_MAPPING.keys()),
                        help="任务标识符")

    # 模型参数
    parser.add_argument("--model_path", type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory/chemechpred/_legacy")
    parser.add_argument("--model_name", type=str, default="updcanoamrxts_cls_to_mech_updcanoamprds_para02",
                        help="模型名称")

    # 数据参数
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/test/random100")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/chemechpred/prediction/random100")

    # 推理参数
    parser.add_argument("--finetuning_type", type=str, default="full")
    parser.add_argument("--template", type=str, default="qwen")
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--num_return_sequences", type=int, default=3)
    parser.add_argument("--output_scores", action="store_true", default=True)
    parser.add_argument("--return_dict_in_generate", action="store_true", default=True)

    # 动态批处理参数
    parser.add_argument("--batch_limit", type=int, default=1,
                        help="每个批次的最大序列数")
    parser.add_argument("--batch_token_size", type=int, default=2000,
                        help="每个批次的最大token数")
    parser.add_argument("--minmax_gap", type=int, default=40,
                        help="批次内允许的最大长度差异")

    # 系统参数
    parser.add_argument("--gpu_threshold", type=float, default=10.0,
                        help="GPU内存阈值(GB)")
    parser.add_argument("--log_dir", type=str, default="./logs/chemechpred",
                        help="日志目录")

    args = parser.parse_args()
    main(args)