import os
import json
from tqdm import tqdm
from rdkit import Chem
from typing import List, Dict, Any
from llamafactory.chat import ChatModel
# python ./application/eval/qwen2/predict.py 2>&1 | tee -a logs/predict/qwen2_0_5b_molecular_transformer_mit_mixed_lora_sft_epoch_3.log
'''
请为我编写Python程序：

已知：
1. data_path=/home/liangtao/DataSets/Chemistry/MolecularTransformer/test, 下有文件MIT_mixed.json
2. MIT_mixed.json中的数据是json数组，形式如下:
[
  {
    "instruction": "PREDICT_PRODUCT:",
    "input": "C 1 C C O C 1 . N # C c 1 c c s c 1 N . O = [N+] ( [O-] ) c 1 c c ( F ) c ( F ) c c 1 F . [H-] . [Na+]",
    "output": "N # C c 1 c c s c 1 N c 1 c c ( F ) c ( F ) c c 1 [N+] ( = O ) [O-]"
  },
  ...
]
3. 相关模型推理代码如下，可以实现beam_size下的推理过程，并得到beam_size个推理结果
```import os

from llamafactory.chat import ChatModel
FILE_NAME='MIT_mixed_test'
MODEL_NAME='qwen2_0_5b_molecular_transformer_mit_mixed_lora_sft_ckpt_60k'
INFER_ARGS = {
    "model_name_or_path": f"/home/liangtao/Development/LLMSpace/LLaMA-Factory/output/{MODEL_NAME}",
    "finetuning_type": "lora",
    "template": "qwen",
    # "num_return_sequences": 10,
    # "infer_dtype": "float16",
    "num_beams": 10,
    "do_sample": False,
    "max_new_tokens": 1000,
}

MESSAGES = [
    {"role": "user", "content": "PREDICT_PRODUCT:\nC 1 C C O C 1 . N # C c 1 c c s c 1 N . O = [N+] ( [O-] ) c 1 c c ( F ) c ( F ) c c 1 F . [H-] . [Na+]"},
]

def chat():
    chat_model = ChatModel(INFER_ARGS)
    responses = chat_model.chat(MESSAGES, num_return_sequences=5, output_scores=True, return_dict_in_generate=True)
    print(responses[0])

if __name__=="__main__":
    chat()
``


4. 上述代码的response如下：
[Response(response_text='C N C ( = O ) c 1 c c c ( Cl ) c ( [N+] ( = O ) [O-] ) c 1', response_length=34, prompt_length=71, sequence_score=0.5440576076507568, finish_reason='stop'), Response(response_text='C N c 1 c c c ( C ( = O ) O ) c c 1 [N+] ( = O ) [O-]', response_length=32, prompt_length=71, sequence_score=0.32601550221443176, finish_reason='stop'), Response(response_text='C N C c 1 c c c ( Cl ) c ( [N+] ( = O ) [O-] ) c 1', response_length=30, prompt_length=71, sequence_score=0.03410578519105911, finish_reason='stop'), Response(response_text='C N c 1 c c ( C ( = O ) O ) c c c 1 Cl', response_length=22, prompt_length=71, sequence_score=0.05452997609972954, finish_reason='stop'), Response(response_text='C N c 1 c c ( C ( = O ) O ) c c c 1 [N+] ( = O ) [O-]', response_length=32, prompt_length=71, sequence_score=0.0023172996006906033, finish_reason='stop')]

要求：
1. 获取所有数据后，对每一条数据拼接成MESSAGES形式，再调用模型进行推理
2. 将得到的结果进行保存，保存路径为output_dir=/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction,
    保存文件为f'{MODEL_NAME}.json'
3. f'{MODEL_NAME}.json'文件内容为json array, 如下：
[
    {
        "id": f"{FILE_NAME}_000000001"
        "instruction": "PREDICT_PRODUCT:",
        "input": "C 1 C C O C 1 . N # C c 1 c c s c 1 N . O = [N+] ( [O-] ) c 1 c c ( F ) c ( F ) c c 1 F . [H-] . [Na+]",
        "label": "N # C c 1 c c s c 1 N c 1 c c ( F ) c ( F ) c c 1 [N+] ( = O ) [O-]"
        "prompt_length": 71
        "output": [
            {
                "text": "C N c 1 c c c ( C ( = O ) O ) c c 1 [N+] ( = O ) [O-]",
                "length": 34,
                "sequence_score": 0.5440576076507568,
                "is_valid": true,
                "is_correct": true
            },
            ...
        ]

    },
    ...
]
output的每个元素中，以sequence_score从大到小排序
id的序号根据在FILE_NAME文件中的顺序递增，从1开始
instruction、input和label分别取自MIT_mixed.json数据文件中的instruction、input和output
prompt_length, length, sequence_score分别来自Response中的prompt_length, response_length, sequence_score字段
is_valid表示该text对应的SMILES是否为有效的SMILES，需要用RDKit工具进行判断，判断前先去除所有空格
is_correct表示text和label是否同时表示一种化合物，可以去除空格和利用RDKit工具进行分析
4. 预测过程中，显示进度条
5. 方法进行适度封装，不要封装过度，从__main__开始执行

'''



# 配置参数
FILE_NAME = 'MIT_mixed_test'
MODEL_NAME = 'qwen2_0_5b_molecular_transformer_mit_mixed_lora_sft_epoch_3'
DATA_PATH = '/home/liangtao/DataSets/Chemistry/MolecularTransformer/test/MIT_mixed.json'
OUTPUT_DIR = '/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.json')

INFER_ARGS = {
    "model_name_or_path": f"/home/liangtao/Development/LLMSpace/LLaMA-Factory/output/{MODEL_NAME}",
    "finetuning_type": "lora",
    "template": "qwen",
    "num_beams": 10,
    "do_sample": False,
    "max_new_tokens": 1000,
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


def process_response(response: List[Dict], label: str) -> List[Dict]:
    """处理模型响应，添加is_valid和is_correct字段，并按score排序"""
    processed = []
    for resp in response:
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
    return processed


def predict_and_save():
    # 加载数据
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    # 初始化模型
    chat_model = ChatModel(INFER_ARGS)

    results = []
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        # 构建消息
        messages = [{
            "role": "user",
            "content": f"{item['instruction']}\n{item['input']}"
        }]

        # 模型推理
        responses = chat_model.chat(
            messages,
            num_return_sequences=10,
            output_scores=True,
            return_dict_in_generate=True
        )

        # 处理结果
        processed_output = process_response(responses, item["output"])

        # 构建结果项
        result_item = {
            "id": f"{FILE_NAME}_{str(idx + 1).zfill(9)}",
            "instruction": item["instruction"],
            "input": item["input"],
            "label": item["output"],
            "prompt_length": responses[0].prompt_length if responses else 0,
            "output": processed_output
        }
        results.append(result_item)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    predict_and_save()