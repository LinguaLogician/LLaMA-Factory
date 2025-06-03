'''
已知：
1. batch调用用例：
```
MESSAGES_LIST = []
MESSAGES_LIST.append([{"role": "user", "content": "PREDICT_PRODUCT:\nC 1 C C O C 1 . Cl . O C C C N 1 C C C C C 1 . O c 1 c c c ( - c 2 n c ( C N 3 C C C C C 3 ) c o 2 ) c c 1 F . c 1 c c c ( P ( c 2 c c c c c 2 ) c 2 c c c c c 2 ) c c 1"},])
MESSAGES_LIST.append([{"role": "user", "content": "PREDICT_PRODUCT:\nC . C C O C ( C ) = O . C c 1 o c ( - c 2 c c c c c 2 ) n c 1 C C O c 1 c c c ( [N+] ( = O ) [O-] ) c n 1 . [Pd]"},])
chat_model = ChatModel(INFER_ARGS)
responses = chat_model.batch_llm_predict(MESSAGES_LIST, num_return_sequences=10, output_scores=True, return_dict_in_generate=True)
print(responses)
```
2. 调用返回：
[[Response(response_text='F c 1 c c ( - c 2 n c ( C N 3 C C C C C 3 ) c o 2 ) c c c 1 O C C C N 1 C C C C C 1', response_length=51, prompt_length=134, sequence_score=0.9745560884475708, finish_reason='stop'), ...], [Response(response_text='C c 1 o c ( - c 2 c c c c c 2 ) n c 1 C C O c 1 c c c ( [N+] ( = O ) [O-] ) c n 1', response_length=50, prompt_length=100, sequence_score=0.37255337834358215, finish_reason='stop'),...]]

请为我修改之前的代码，保持原来过程总体不变，将调用改成使用chat_model.batch_llm_predict形式，可以通过设置从制定的数据文件中获取文件后以batch_size大小批量调用

如果可以，使用DataLoader形式加载数据进行迭代
'''

import os
import json
from tqdm import tqdm
from rdkit import Chem
from typing import List, Dict, Any, Iterator
from torch.utils.data import Dataset, DataLoader
from llamafactory.chat import ChatModel

# 配置参数
FILE_NAME = 'MIT_mixed_test'
MODEL_NAME = 'qwen2_0_5b_molecular_transformer_mit_mixed_lora_sft_epoch_3'
DATA_PATH = '/home/liangtao/DataSets/Chemistry/MolecularTransformer/test/MIT_mixed.json'
OUTPUT_DIR = '/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_batch.json')

INFER_ARGS = {
    "model_name_or_path": f"/home/liangtao/Development/LLMSpace/LLaMA-Factory/output/{MODEL_NAME}",
    "finetuning_type": "lora",
    "template": "qwen",
    "num_beams": 10,
    "do_sample": False,
    "max_new_tokens": 1000,
}


class ChemistryDataset(Dataset):
    """化学数据集加载器"""

    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"],
            "id": f"{FILE_NAME}_{str(idx + 1).zfill(9)}"
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


def batch_predict_and_save(batch_size: int = 32):
    """批量预测并保存结果"""
    # 加载数据集
    dataset = ChemistryDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    chat_model = ChatModel(INFER_ARGS)

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
            num_return_sequences=10,
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
                "prompt_length": batch_responses[idx][0].prompt_length if batch_responses[idx] else 0,
                "output": processed_outputs[idx]
            }
            results.append(result_item)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    batch_predict_and_save(batch_size=1)

# python ./application/eval/qwen2/predict_batch.py 2>&1 | tee -a logs/predict/qwen2_0_5b_molecular_transformer_mit_mixed_lora_sft_epoch_3_demo_batch.log
