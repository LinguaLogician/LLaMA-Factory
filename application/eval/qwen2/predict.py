# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: predict.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/6/3 15:01
import gc
import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from rdkit import Chem
from typing import List, Dict, Any
from torch.utils.data import Dataset
from llamafactory.chat import ChatModel

'''
继续为我优化：
1. 设置一个参数batch_token_size, 表示一个batch中，最多有的token_id的总长度
2. 设置一个参数minmax_gap, 表示一个batch中，序列的最长和最短的的token_id长度不能超过minmax_gap
3. DataLoader在组装数据时，一个batch受到batch_token_size和minmax_gap的限制, 将原参数batch_size改为batch_limit, 表示一个batch中最大可以接受的序列个数
4. 将原来的固定batch_size改为根据上述约束而进行的动态长度batch
5. 将最终从数据集predict后得到的结果根据id进行排序，保持和原来顺序一致
6. 将程序的代码整理后，对变量进行抽取，抽取的变量包括：
	INFER_ARGS中的所有变量
	num_return_sequences
	output_scores
	return_dict_in_generate
	batch_limit
	FILE_NAME
	MODEL_NAME
	DATA_PATH
	OUTPUT_DIR
	OUTPUT_FILE
可以改变原先变量名，使得程序通过arg parser设置变量，并可以通过外部shell执行脚本传入变量
'''


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
                # "id": f"{file_prefix}_{str(idx + 1).zfill(9)}",
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


def process_responses(responses: List, reference: str) -> List[Dict]:
    """处理模型响应"""
    processed = []
    for resp in responses:
        text = resp.response_text
        item = {
            "text": text,
            "length": resp.response_length,
            "sequence_score": resp.sequence_score,
            "is_valid": validate_smiles(text),
            "is_correct": compare_molecules(text, reference)
        }
        processed.append(item)

    # 按分数降序排列
    processed.sort(key=lambda x: x["sequence_score"], reverse=True)
    return processed


def main(args):
    checkpoints = []
    if not args.checkpoints or args.checkpoints=="":
        checkpoints.append(Path(args.model_path) / args.model_name)
    elif args.checkpoints == "all":
        checkpoints = [p for p in (Path(args.model_path) / args.model_name).rglob('*')
                       if p.is_dir() and 'ckpt' in p.name]
    else:
        checkpoints = [(Path(args.model_path) / args.model_name / ckpt.strip())
                       for ckpt in args.checkpoints.split(',')]

    infer_args = {
        "model_name_or_path": str(checkpoints[0]),
        "finetuning_type": args.finetuning_type,
        "template": args.template,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
    }
    chat_model = ChatModel(infer_args)
    tokenizer = chat_model.engine.tokenizer
    # Load dataset once (outside the loop)
    data_path = os.path.join(args.data_dir, args.data_file)
    dataset = DynamicBatchDataset(str(data_path), tokenizer)
    batches = dataset.create_batches(args.batch_limit, args.batch_token_size, args.minmax_gap)

    for ckpt in checkpoints:
        if 'chat_model' in locals():
            del chat_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()
        infer_args.update(model_name_or_path=str(ckpt))
        # try:
        chat_model = ChatModel(infer_args)
        tokenizer = chat_model.engine.tokenizer

        results = [None] * len(dataset)
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
                processed = process_responses(responses, sample["output"])
                results[sample["original_idx"]] = {
                    "id": sample["id"],
                    "instruction": sample["instruction"],
                    "input": sample["input"],
                    "label": sample["output"],
                    "prompt_length": sample["length"],
                    "output": processed
                }

        results = [r for r in results if r is not None]
        ckpt_name = ckpt.name if ckpt.name.startswith("ckpt") else "ckptlast"
        output_path = os.path.join(args.output_dir, f"{args.model_name}_{ckpt_name}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Prediction completed for {args.model_name} - {ckpt_name}. Results saved to {output_path}")
        # finally:
        #     if 'chat_model' in locals():
        #         del chat_model
        #         if torch.cuda.is_available():
        #             torch.cuda.empty_cache()
        #     gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecular Transformer Batch Prediction")

    parser.add_argument("--model_path", type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory/output")
    parser.add_argument("--model_name", type=str,
                        default="qwen205_moltrans_mit_mixed_space_lora_para1")
    parser.add_argument("--checkpoints", type=str, default="all") # or "ckpt20k,ckpt40k"
    parser.add_argument("--data_file", type=str, default="MIT_mixed.json")
    parser.add_argument("--data_dir", type=str,
                        default="/home/liangtao/DataSets/Chemistry/MolecularTransformer/space/test/random100")
    parser.add_argument("--output_dir", type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction/"
                                "mit_mixed_space_test_random100")
    # Inference parameters
    parser.add_argument("--finetuning_type", type=str, default="lora")
    parser.add_argument("--template", type=str, default="qwen")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--output_scores", action="store_true", default=True)
    parser.add_argument("--return_dict_in_generate", action="store_true", default=True)

    # Dynamic batching parameters
    parser.add_argument("--batch_limit", type=int, default=1,
                        help="Maximum number of sequences in a batch")
    parser.add_argument("--batch_token_size", type=int, default=500,
                        help="Maximum total token size per batch")
    parser.add_argument("--minmax_gap", type=int, default=20,
                        help="Maximum allowed length difference within a batch")
    main(parser.parse_args())
# python ./application/eval/qwen2/predict.py 2>&1 | tee -a logs/predict/qwen2_0_5b_molecular_transformer_mit_mixed_lora_sft_ckpt60k.log
