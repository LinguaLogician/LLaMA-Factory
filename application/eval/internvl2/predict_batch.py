# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: predict_batch.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/3 8:07
# https://chat.deepseek.com/a/chat/s/d2111ac9-cbbc-445a-a4e4-e26c2743cc4c

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from rdkit import Chem
import torch
from torch.utils.data import Dataset, DataLoader

from llamafactory.chat import ChatModel

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChemicalDataset(Dataset):
    """化学数据数据集类"""

    def __init__(self, test_file_path: str):
        """
        初始化数据集

        Args:
            test_file_path: 测试文件路径
        """
        self.data = self._load_data(test_file_path)

    def _load_data(self, test_file_path: str) -> List[Dict[str, Any]]:
        """加载测试数据"""
        data = []
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个数据项"""
        return self.data[idx]

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        批处理函数

        Returns:
            Tuple: (原始数据批次, 消息列表批次)
        """
        messages_batch = []
        original_data_batch = []

        for item in batch:
            # 保存原始数据
            original_data_batch.append(item)

            # 创建消息格式
            messages = [{
                "role": "user",
                "images": [item["image"]],
                "content": item["conversations"][0]["value"]
            }]
            messages_batch.append(messages)

        return original_data_batch, messages_batch


def is_valid_smiles(smiles: str) -> bool:
    """检查SMILES字符串是否有效"""
    try:
        # 去除空格
        smiles = smiles.strip().replace(" ", "")
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def are_same_molecule(smiles1: str, smiles2: str) -> bool:
    """检查两个SMILES字符串是否表示同一分子"""
    try:
        # 去除空格
        smiles1 = smiles1.strip().replace(" ", "")
        smiles2 = smiles2.strip().replace(" ", "")

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return False

        # 规范化SMILES进行比较
        can_smiles1 = Chem.MolToSmiles(mol1, canonical=True)
        can_smiles2 = Chem.MolToSmiles(mol2, canonical=True)

        return can_smiles1 == can_smiles2
    except:
        return False


def process_responses(responses: Any, original_data: Dict[str, Any]) -> Dict[str, Any]:
    """处理模型响应"""
    # 提取人类指令和GPT标签
    human_message = next((conv for conv in original_data["conversations"] if conv["from"] == "human"), None)
    gpt_message = next((conv for conv in original_data["conversations"] if conv["from"] == "gpt"), None)

    instruction = human_message["value"] if human_message else ""
    label = gpt_message["value"] if gpt_message else ""

    # 处理每个响应
    output_list = []

    # 确保responses是可迭代的
    if not hasattr(responses, '__iter__'):
        responses = [responses]

    # 按sequence_score排序
    sorted_responses = sorted(responses, key=lambda x: getattr(x, 'sequence_score', 0), reverse=True)

    for resp in sorted_responses:
        output_item = {
            "text": getattr(resp, 'response_text', ''),
            "length": getattr(resp, 'response_length', 0),
            "sequence_score": getattr(resp, 'sequence_score', 0),
            "is_valid": is_valid_smiles(getattr(resp, 'response_text', '')),
            "is_correct": are_same_molecule(getattr(resp, 'response_text', ''), label)
        }
        output_list.append(output_item)

    # 构建结果项
    result_item = {
        "id": original_data["id"],
        "instruction": instruction,
        "input": "",
        "label": label,
        "prompt_length": getattr(sorted_responses[0], 'prompt_length', 0) if sorted_responses else 0,
        "output": output_list
    }

    return result_item


def main():

    # 构建完整路径
    model_path = os.path.join(args.model_location, args.model_name)
    test_file_path = os.path.join(args.test_path, args.test_file)
    output_file = os.path.join(args.output_dir, f"{args.model_name}.json")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置推理参数
    infer_args = {
        "model_name_or_path": model_path,
        "finetuning_type": args.finetuning_type,
        "template": args.template,
        "num_beams": args.num_beams,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "trust_remote_code": args.trust_remote_code,
    }

    # 创建数据集和数据加载器
    logger.info(f"Loading test data from {test_file_path}")
    dataset = ChemicalDataset(test_file_path)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=False
    )

    logger.info(f"Dataset size: {len(dataset)}, Number of batches: {len(dataloader)}")

    # 初始化模型
    logger.info("Initializing chat model")
    chat_model = ChatModel(infer_args)

    results = []

    # 进行批量推理
    logger.info(f"Starting inference with batch size {args.batch_size}")

    with torch.no_grad():
        for batch_idx, (original_batch, messages_batch) in enumerate(tqdm(dataloader, desc="Processing batches")):
            try:
                # 批量预测
                responses_batch = chat_model.batch_lmm_predict(
                    messages_batch,
                    num_return_sequences=args.num_return_sequences,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                # 处理每个样本的响应
                for i, (original_data, responses) in enumerate(zip(original_batch, responses_batch)):
                    result_item = process_responses(responses, original_data)
                    results.append(result_item)

                # 每处理完10个批次打印一次进度
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1} batches, {len(results)} samples")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                # 记录错误但继续处理
                continue

    # 保存结果
    logger.info(f"Saving {len(results)} results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Inference completed successfully")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run inference on chemical data")

    # 模型相关参数
    parser.add_argument("--model_location", type=str, default="/mnt/d/ChemicalFactory/output/",
                        help="Path to model directory")
    parser.add_argument("--model_name", type=str, default="internvl21_chemicals_retrosyn_full_para01",
                        help="Name of the model")

    # 推理参数
    parser.add_argument("--finetuning_type", type=str, default="full",
                        help="Finetuning type")
    parser.add_argument("--template", type=str, default="intern_vl",
                        help="Template for the model")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.95,
                        help="Temperature for sampling")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--trust_remote_code", type=bool, default=True,
                        help="Whether to trust remote code")

    # 数据相关参数
    parser.add_argument("--test_path", type=str,
                        default="/mnt/e/Development/ChemistrySpace/ChemProphet/data/chemicals/retrosyn/",
                        help="Path to test data directory")
    parser.add_argument("--test_file", type=str, default="retrosyn_test_internvl2.jsonl",
                        help="Test file name")

    # 推理配置
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for inference")
    parser.add_argument("--num_return_sequences", type=int, default=5,
                        help="Number of return sequences")
    parser.add_argument("--output_dir", type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction",
                        help="Output directory for results")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--shuffle", type=bool, default=True,
                        help="Whether to shuffle the data")

    args = parser.parse_args()
    main()