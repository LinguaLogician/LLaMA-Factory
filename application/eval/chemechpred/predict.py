# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: predict.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/21 9:28
# https://chat.deepseek.com/a/chat/s/e2277faf-eb05-4af3-8ffe-031dd0c3b00d

# -*- coding: utf-8 -*-
import gc
import os
import json
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import Dataset
from llamafactory.chat import ChatModel

# 任务定义
TASKS = {
    "AMRXTS_TO_AMPRDS": {
        "input_fields": ["AMRXTS"],
        "output_fields": ["AMPRDS"]
    },
    "AMRXTS_TO_MECH_AMPRDS": {
        "input_fields": ["AMRXTS"],
        "output_fields": ["MECH", "AMPRDS"]
    },
    "AMRXTS_TO_CLS_AMPRDS": {
        "input_fields": ["AMRXTS"],
        "output_fields": ["CLS", "AMPRDS"]
    },
    "AMRXTS_TO_CLS_MECH_AMPRDS": {
        "input_fields": ["AMRXTS"],
        "output_fields": ["CLS", "MECH", "AMPRDS"]
    },
    "AMRXTS_MECH_TO_AMPRDS": {
        "input_fields": ["AMRXTS", "MECH"],
        "output_fields": ["AMPRDS"]
    },
    "AMRXTS_CLS_TO_AMPRDS": {
        "input_fields": ["AMRXTS", "CLS"],
        "output_fields": ["AMPRDS"]
    },
    "AMRXTS_MECH_CLS_TO_AMPRDS": {
        "input_fields": ["AMRXTS", "MECH", "CLS"],
        "output_fields": ["AMPRDS"]
    }
}


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


def validate_smiles(smiles: str) -> bool:
    """验证SMILES有效性"""
    try:
        return Chem.MolFromSmiles(smiles.replace(" ", "")) is not None
    except:
        return False


def compare_molecules(smiles1: str, smiles2: str) -> bool:
    """比较两个分子是否相同"""
    try:
        mol1 = Chem.MolFromSmiles(smiles1.replace(" ", ""))
        mol2 = Chem.MolFromSmiles(smiles2.replace(" ", ""))
        return mol1 is not None and mol2 is not None and Chem.MolToInchi(mol1) == Chem.MolToInchi(mol2)
    except:
        return False


def parse_model_output(text: str, task_id: str) -> Dict[str, Any]:
    """
    解析模型输出，提取各个字段
    """
    task_config = TASKS[task_id]
    output_fields = task_config["output_fields"]

    result = {"text": text, "is_task_matched": True, "is_resolved_correct": True}

    # 按行分割输出
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # 检查输出是否包含所有必需的字段
    found_fields = []
    field_values = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        if line in output_fields and line not in found_fields:
            field_name = line
            found_fields.append(field_name)

            # 获取字段值（下一行）
            if i + 1 < len(lines) and lines[i + 1] not in output_fields:
                field_values[field_name] = lines[i + 1]
                i += 2
            else:
                field_values[field_name] = ""
                i += 1
        else:
            i += 1

    # 检查是否所有必需字段都被找到
    missing_fields = set(output_fields) - set(found_fields)
    if missing_fields:
        result["is_task_matched"] = False
        result["missing_fields"] = list(missing_fields)

    # 为每个字段添加结果
    for field in output_fields:
        if field in field_values:
            result[field.lower()] = field_values[field]
        else:
            result[field.lower()] = ""
            result["is_resolved_correct"] = False

    return result


def evaluate_output(parsed_output: Dict[str, Any], label_parsed: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    评估解析后的输出
    """
    task_config = TASKS[task_id]
    output_fields = task_config["output_fields"]

    evaluation = parsed_output.copy()
    evaluation["is_correct_totally"] = True

    for field in output_fields:
        field_lower = field.lower()
        pred_value = parsed_output.get(field_lower, "")
        true_value = label_parsed.get(field_lower, "")

        # 对于AMPRDS字段，需要验证SMILES和比较分子
        if field == "AMPRDS":
            evaluation["is_valid_amprds"] = validate_smiles(pred_value)
            evaluation["is_correct_amprds"] = compare_molecules(pred_value, true_value) if evaluation[
                "is_valid_amprds"] else False

            if not evaluation["is_correct_amprds"]:
                evaluation["is_correct_totally"] = False

        # 对于其他字段（CLS, MECH），进行字符串比较
        elif field in ["CLS", "MECH"]:
            evaluation[f"is_correct_{field_lower}"] = (pred_value == true_value)
            if not evaluation[f"is_correct_{field_lower}"]:
                evaluation["is_correct_totally"] = False

    return evaluation


def process_responses(responses: List, reference: str, task_id: str) -> List[Dict]:
    """处理模型响应"""
    # 首先解析参考输出
    label_parsed = parse_model_output(reference, task_id)

    processed = []
    for resp in responses:
        text = resp.response_text
        parsed_output = parse_model_output(text, task_id)

        # 添加模型生成的基本信息
        item = {
            "text": text,
            "length": resp.response_length,
            "sequence_score": resp.sequence_score,
        }

        # 添加解析后的字段
        item.update(parsed_output)

        # 添加评估结果
        evaluated = evaluate_output(parsed_output, label_parsed, task_id)
        item.update({k: v for k, v in evaluated.items() if k not in item})

        processed.append(item)

    # 按分数降序排列
    processed.sort(key=lambda x: x["sequence_score"], reverse=True)
    return processed


def wait_for_gpu_memory(threshold: int, check_interval: int = 30):
    """等待GPU内存可用"""
    if not torch.cuda.is_available():
        return

    while True:
        torch.cuda.empty_cache()
        free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        free_memory_mb = free_memory / 1024 / 1024

        if free_memory_mb >= threshold:
            logging.info(f"GPU memory available: {free_memory_mb:.2f} MB (threshold: {threshold} MB)")
            return

        logging.info(f"Waiting for GPU memory... Current: {free_memory_mb:.2f} MB, Required: {threshold} MB")
        time.sleep(check_interval)


def main(args):
    # 设置日志
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

    # 检查GPU内存
    if args.gpu_threshold > 0:
        wait_for_gpu_memory(args.gpu_threshold)

    # 设置任务相关路径
    task_id_lower = args.task_id.lower()
    data_file = f"{task_id_lower}_test.json"
    data_path = os.path.join(args.data_dir, data_file)
    output_dir = os.path.join(args.output_dir, task_id_lower)
    os.makedirs(output_dir, exist_ok=True)

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

    results = [None] * len(dataset)

    # 处理每个batch
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
            # 解析参考输出
            reference_parsed = parse_model_output(sample["output"], args.task_id)
            label_info = {
                "text": sample["output"],
                **{k: v for k, v in reference_parsed.items() if
                   k not in ["text", "is_task_matched", "is_resolved_correct"]}
            }

            processed = process_responses(responses, sample["output"], args.task_id)
            results[sample["original_idx"]] = {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "input": sample["input"],
                "prompt_length": sample["length"],
                "label": label_info,
                "output": processed
            }

    # 保存结果
    output_file = os.path.join(output_dir, f"{args.model_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Evaluation completed for {args.model_name} on task {args.task_id}. Results saved to {output_file}")

    # 清理资源
    del chat_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chemical Mechanism Prediction Evaluation")

    # 模型和任务参数
    parser.add_argument("--model_path", type=str, default="/mnt/e/Development/LLMSpace/LLaMA-Factory/output")
    parser.add_argument("--model_name", type=str, default="qwen205_amrxts_to_cls_mech_amprds_para01", help="Name of the model to evaluate")
    parser.add_argument("--task_id", type=str, default="AMRXTS_TO_CLS_MECH_AMPRDS", choices=list(TASKS.keys()), help="Task identifier")

    # 数据路径参数
    parser.add_argument("--data_dir", type=str, default="/mnt/e/DataSets/Chemistry/ChemicalMechanism/test")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/chemechpred/prediction")

    # Inference parameters
    parser.add_argument("--finetuning_type", type=str, default="full")
    parser.add_argument("--template", type=str, default="qwen")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--output_scores", action="store_true", default=True)
    parser.add_argument("--return_dict_in_generate", action="store_true", default=True)

    # Dynamic batching parameters
    parser.add_argument("--batch_limit", type=int, default=1, help="Maximum number of sequences in a batch")
    parser.add_argument("--batch_token_size", type=int, default=500, help="Maximum total token size per batch")
    parser.add_argument("--minmax_gap", type=int, default=20, help="Maximum allowed length difference within a batch")

    # GPU 和日志参数
    parser.add_argument("--gpu_threshold", type=int, default=1024, help="Minimum GPU memory required (MB)")
    parser.add_argument("--log_file", type=str,
                        default=f"./logs/chemechpred/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    args = parser.parse_args()
    main(args)
