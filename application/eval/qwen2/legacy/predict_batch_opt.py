'''
在使用上述代码过程中，发现最后batchsize下的的消耗时间反而更长，估计可能和batch中大小不一致导致的多余padding部分参与计算有关，请为我做出优化
已知
1. 可以通过chat_model.engine.tokenizer获得分词器
可否帮我再加载数据过程中，根据分此后的prompt_ids长度进行排序，使得每一个batch中的length尽量接近
'''

import os
import json
from tqdm import tqdm
from rdkit import Chem
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from llamafactory.chat import ChatModel

# 配置参数
FILE_NAME = 'MIT_mixed_test'
MODEL_NAME = 'qwen205_moltrans_mit_separated_space_lora_para1_epoch3'
DATA_PATH = '/home/liangtao/DataSets/Chemistry/MolecularTransformer/space/test/MIT_separated.json'
OUTPUT_DIR = '/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.json')

INFER_ARGS = {
    "model_name_or_path": f"/home/liangtao/Development/LLMSpace/LLaMA-Factory/output/{MODEL_NAME}",
    "finetuning_type": "lora",
    "template": "qwen",
    "num_beams": 5,
    "do_sample": False,
    "max_new_tokens": 1000,
}


class ChemistryDataset(Dataset):
    """化学数据集加载器，支持按长度排序"""

    def __init__(self, data_path: str, tokenizer):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        # 预处理数据并计算token长度
        self.data = []
        for idx, item in enumerate(raw_data):
            prompt = f"{item['instruction']}\n{item['input']}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            self.data.append({
                "id": f"{FILE_NAME}_{str(idx + 1).zfill(9)}",
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
                "length": len(prompt_ids)
            })

        # 按token长度排序
        self.data.sort(key=lambda x: x["length"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "id": item["id"],
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"],
            "length": item["length"]
        }


def is_valid_smiles(smiles: str) -> bool:
    """检查SMILES字符串是否有效"""
    try:
        mol = Chem.MolFromSmiles(smiles.replace(" ", ""))
        return mol is not None
    except:
        return False


def are_same_molecule(smiles1: str, smiles2: str) -> bool:
    """检查两个SMILES字符串是否表示同一分子"""
    try:
        mol1 = Chem.MolFromSmiles(smiles1.replace(" ", ""))
        mol2 = Chem.MolFromSmiles(smiles2.replace(" ", ""))
        if mol1 is None or mol2 is None:
            return False
        return Chem.MolToInchi(mol1) == Chem.MolToInchi(mol2)
    except:
        return False


def process_batch_responses(batch_responses: List[List[Any]], labels: List[str]) -> List[List[Dict]]:
    """处理批量响应结果"""
    processed_batch = []
    for responses, label in zip(batch_responses, labels):
        processed = []
        for resp in responses:
            text = resp.response_text
            item = {
                "text": text,
                "length": resp.response_length,
                "sequence_score": resp.sequence_score,
                "is_valid": is_valid_smiles(text),
                "is_correct": are_same_molecule(text, label)
            }
            processed.append(item)

        # 按sequence_score降序排序
        processed.sort(key=lambda x: x["sequence_score"], reverse=True)
        processed_batch.append(processed)
    return processed_batch


def collate_fn(batch):
    """自定义collate函数，保持原始顺序"""
    return {
        "id": [item["id"] for item in batch],
        "instruction": [item["instruction"] for item in batch],
        "input": [item["input"] for item in batch],
        "output": [item["output"] for item in batch],
        "length": [item["length"] for item in batch]
    }


def batch_predict_and_save(batch_size: int = 32):
    """批量预测并保存结果"""
    # 初始化模型和分词器
    chat_model = ChatModel(INFER_ARGS)
    tokenizer = chat_model.engine.tokenizer

    # 加载数据集并按长度排序
    dataset = ChemistryDataset(DATA_PATH, tokenizer)

    # 创建DataLoader，禁用自动批处理，使用自定义collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    results = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        # 构建批量消息
        batch_messages = [[{
            "role": "user",
            "content": f"{instruction}\n{input_text}"
        }] for instruction, input_text in zip(batch["instruction"], batch["input"])]

        # 批量推理
        batch_responses = chat_model.batch_llm_predict(
            batch_messages,
            num_return_sequences=5,
            output_scores=True,
            return_dict_in_generate=True
        )

        # 处理批量结果
        processed_outputs = process_batch_responses(batch_responses, batch["output"])

        # 构建结果项
        for idx in range(len(batch["id"])):
            result_item = {
                "id": batch["id"][idx],
                "instruction": batch["instruction"][idx],
                "input": batch["input"][idx],
                "label": batch["output"][idx],
                "prompt_length": batch["length"][idx],
                "output": processed_outputs[idx]
            }
            results.append(result_item)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    batch_predict_and_save(batch_size=2)

# python ./application/eval/qwen2/predict_batch_opt.py 2>&1 | tee -a logs/predict/qwen205_moltrans_mit_separated_space_lora_para1_epoch3.log
