# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: score.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/6/1 20:22

'''
请为我编写Python程序，
已知：
REDICTION_FILE=qwen2_0_5b_molecular_transformer_mit_mixed_lora_sft_epoch_3
1. prediction_dir=/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction, 其中有文件名为f'{REDICTION_FILE}.json'的文件
2. f'{REDICTION_FILE}.json'文件的内容形式如下：
[
  {
    "id": "MIT_mixed_test_000000001",
    "instruction": "PREDICT_PRODUCT:",
    "input": "C 1 C C O C 1 . N # C c 1 c c s c 1 N . O = [N+] ( [O-] ) c 1 c c ( F ) c ( F ) c c 1 F . [H-] . [Na+]",
    "label": "N # C c 1 c c s c 1 N c 1 c c ( F ) c ( F ) c c 1 [N+] ( = O ) [O-]",
    "prompt_length": 95,
    "output": [
      {
        "text": "N # C c 1 c c s c 1 N c 1 c c ( F ) c ( F ) c c 1 [N+] ( = O ) [O-]",
        "length": 41,
        "sequence_score": 0.9956136345863342,
        "is_valid": true,
        "is_correct": true
      },
      ...
    ]
  },
  ...
]

一个文件中是一个json array, 有一定数量的元素，每个元素的output有beam_size个结果，结果以sequence_score从大到小排序。
is_correct表示预测是否正确
is_valid表示预测是否有效
3. output_dir=/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/scores
4. topk的Accuracy表示，对每个预测，前k个预测结果存在至少一个is_correct为true预测占所有预测数的比率
5. topk的有效率表示，对每个预测，前k个预测结果中is_valid为true的结果数总和占所有预测结果数(即预测数*k)的比率
要求：
1. 根据is_correct计算topk的Accracy，k=1,2,3,5,10
2. 根据is_valid计算topk的有效率，k=1,2,3,5,10
3. 计算结果保存到scores.txt中，其中不要有汉字，记录：
  预测样例数和beam_size数目
  topk的Accuracy, 有效率
  REDICTION_FILE名称
4. 预测过程中，显示进度条
5. 方法进行适度封装，不要封装过度，从__main__开始执行
'''

import json
from pathlib import Path
from tqdm import tqdm


def calculate_metrics(prediction_file_path, output_dir):
    # Load prediction data
    with open(prediction_file_path, 'r') as f:
        data = json.load(f)

    if not data:
        raise ValueError("Prediction file is empty")

    total_samples = len(data)
    beam_size = len(data[0]['output']) if total_samples > 0 else 0

    # Initialize counters
    topk_acc_counts = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0}
    topk_valid_counts = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0}

    # Process each sample
    for sample in tqdm(data, desc="Processing predictions"):
        outputs = sample['output']

        # Calculate top-k accuracy
        for k in topk_acc_counts:
            if k > beam_size:
                continue
            top_k_outputs = outputs[:k]
            if any(output['is_correct'] for output in top_k_outputs):
                topk_acc_counts[k] += 1

        # Calculate top-k validity
        for k in topk_valid_counts:
            if k > beam_size:
                continue
            top_k_outputs = outputs[:k]
            valid_count = sum(1 for output in top_k_outputs if output['is_valid'])
            topk_valid_counts[k] += valid_count

    # Calculate accuracy rates
    accuracy_rates = {
        k: count / total_samples
        for k, count in topk_acc_counts.items()
        if k <= beam_size
    }

    # Calculate validity rates
    validity_rates = {
        k: count / (total_samples * k)
        for k, count in topk_valid_counts.items()
        if k <= beam_size
    }

    return {
        'total_samples': total_samples,
        'beam_size': beam_size,
        'accuracy_rates': accuracy_rates,
        'validity_rates': validity_rates
    }


def save_results(results, output_dir, prediction_file_name):
    output_path = Path(output_dir) / f'{prediction_file_name}.txt'

    with open(output_path, 'w') as f:
        f.write(f"PREDICTION_FILE: {prediction_file_name}\n")
        f.write(f"Total samples: {results['total_samples']}\n")
        f.write(f"Beam size: {results['beam_size']}\n\n")

        f.write("Top-k Accuracy:\n")
        for k, rate in results['accuracy_rates'].items():
            f.write(f"Top-{k}: {rate:.4f}\n")

        f.write("\nTop-k Validity Rate:\n")
        for k, rate in results['validity_rates'].items():
            f.write(f"Top-{k}: {rate:.4f}\n")


if __name__ == '__main__':
    PREDICTION_FILE = "qwen2_0_5b_molecular_transformer_mit_mixed_augm_lora_sft_epoch_3"
    prediction_dir = "/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction"
    output_dir = "/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/scores"

    prediction_file_path = Path(prediction_dir) / f"{PREDICTION_FILE}.json"

    try:
        results = calculate_metrics(prediction_file_path, output_dir)
        save_results(results, output_dir, PREDICTION_FILE)
        print("Metrics calculation completed successfully.")
    except Exception as e:
        print(f"Error: {str(e)}")