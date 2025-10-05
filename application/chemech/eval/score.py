# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: score.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/21 11:01
# https://chat.deepseek.com/a/chat/s/e2277faf-eb05-4af3-8ffe-031dd0c3b00d

# -*- coding: utf-8 -*-
import os
import json
import argparse
import logging
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm

# 任务定义
TASKS = {
    "UPDCANOAMRXTS_CLS_TO_MECH_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS+CLS->MECH+UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS"],
        "output_fields": ["MECH", "UPDCANOAMPRDS"],
        "metrics": ["mech",  "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    },
    "UPDCANOAMRXTS_CLS_TO_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS+CLS->UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS", "CLS"],
        "output_fields": ["UPDCANOAMPRDS"],
        "metrics": [ "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    },
    "UPDCANOAMRXTS_MECH_CLS_TO_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS+MECH+CLS->UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS", "MECH", "CLS"],
        "output_fields": ["UPDCANOAMPRDS"],
        "metrics": [ "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    },
    "UPDCANOAMRXTS_MECH_TO_CLS_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS+MECH->CLS+UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS", "MECH"],
        "output_fields": ["CLS", "UPDCANOAMPRDS"],
        "metrics": ["cls",  "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    },
    "UPDCANOAMRXTS_TO_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS->UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS"],
        "output_fields": ["UPDCANOAMPRDS"],
        "metrics": [ "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    },
    "UPDCANOAMRXTS_TO_MECH_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS->MECH+UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS"],
        "output_fields": ["MECH", "UPDCANOAMPRDS"],
        "metrics": ["mech",  "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    },
    "UPDCANOAMRXTS_TO_CLS_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS->CLS+UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS"],
        "output_fields": ["CLS", "UPDCANOAMPRDS"],
        "metrics": ["cls",  "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    },
    "UPDCANOAMRXTS_TO_CLS_MECH_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS->CLS+MECH+UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS"],
        "output_fields": ["CLS", "MECH", "UPDCANOAMPRDS"],
        "metrics": ["cls", "mech",  "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    },
    "UPDCANOAMRXTS_MECH_TO_UPDCANOAMPRDS": {
        "task_tag": "UPD.CANO.AM.RXTS+MECH->UPD.CANO.AM.PRDS",
        "input_fields": ["UPDCANOAMRXTS", "MECH"],
        "output_fields": ["UPDCANOAMPRDS"],
        "metrics": [ "upd_cano_am_prds",  "upd_am_prds", "totally", "resolved", "matched"]
    }
}

TASK_TAG_DICT = {
    task["task_tag"]: task_id
    for task_id, task in TASKS.items()
}
PROPERTY_DICT = {
    "resolved": "is_resolved_correct",
    "matched": "is_task_matched",
}


def calculate_metrics(data: List[Dict], task_id: str, num_return_sequences: int) -> Dict[str, Any]:
    """
    计算各种评估指标
    """
    task_config = TASKS[task_id]
    metrics_config = task_config["metrics"]
    n_samples = len(data)

    # 初始化结果字典
    results = {
        "file": "",
        "task": task_id,
        "count": n_samples,
        "num_return_sequences": num_return_sequences
    }

    # 为每个指标初始化数据结构
    for metric in metrics_config:
        if metric == "upd_cano_am_prds":
            results[metric] = {
                "AccTopK": defaultdict(float),
                "AccKth": defaultdict(float),
                "ValTopK": defaultdict(float),
                "ValKth": defaultdict(float)
            }
        elif metric == "upd_am_prds":
            results[metric] = {
                "AccTopK": defaultdict(float),
                "AccKth": defaultdict(float)
            }
        elif metric in ["totally", "resolved", "matched"]:
            results[metric] = {
                "TopK": defaultdict(float),
                "Kth": defaultdict(float)
            }
            pass
        else:
            results[metric] = {
                "AccTopK": defaultdict(float),
                "AccKth": defaultdict(float)
            }

    # 为每个样本和每个k值计算指标
    for k in range(1, num_return_sequences + 1):
        # 初始化计数器
        counters = {
            metric: {
                "topk_correct": 0,  # 前k个中至少有一个正确的样本数
                "kth_correct": 0,  # 第k个正确的样本数
                "topk_total": 0,  # 前k个中正确的总数（用于比率计算）
                "topk_valid": 0,
                "kth_valid": 0
            }
            for metric in metrics_config
        }

        # 处理每个样本
        for sample in tqdm(data, desc=f"Calculating metrics for k={k}"):
            outputs = sample.get("output", [])

            # 计算每个指标的统计
            for metric in metrics_config:
                metric_key = f"is_correct_{metric}" if metric != "totally" else "is_correct_totally"
                if metric in ["resolved", "matched"]:
                    metric_key = PROPERTY_DICT.get( metric, None)

                # 检查前k个输出
                topk_correct = any(output.get(metric_key, False) for output in outputs[:k])
                counters[metric]["topk_correct"] += int(topk_correct)

                # 检查第k个输出（如果存在）
                if k <= len(outputs):
                    kth_correct = outputs[k - 1].get(metric_key, False)
                    counters[metric]["kth_correct"] += int(kth_correct)

                    # 对于topk_total，统计前k个中正确的数量
                    correct_in_topk = sum(1 for output in outputs[:k] if output.get(metric_key, False))
                    counters[metric]["topk_total"] += correct_in_topk

                # 对于amprds，还需要计算有效性
                if metric == "upd_cano_am_prds":
                    topk_valid = sum(output.get("is_valid_upd_cano_am_prds", False) for output in outputs[:k])
                    counters[metric]["topk_valid"] += int(topk_valid)

                    if k <= len(outputs):
                        kth_valid = outputs[k - 1].get("is_valid_upd_cano_am_prds", False)
                        counters[metric]["kth_valid"] += int(kth_valid)

        # 计算比率并存储结果
        for metric in metrics_config:

            if metric in ["totally", "resolved", "matched"]:
                results[metric]["TopK"][f"K={k}"] = counters[metric]["topk_total"] / (n_samples*k) if n_samples > 0 else 0
                results[metric]["Kth"][f"K={k}"] = counters[metric]["kth_correct"] / n_samples if n_samples > 0 else 0
            # 对于amprds，还需要计算有效性指标
            else:
                if metric == "upd_cano_am_prds":
                    results[metric]["ValTopK"][f"K={k}"] = counters[metric][
                                                               "topk_valid"] / (n_samples * k) if n_samples > 0 else 0
                    results[metric]["ValKth"][f"K={k}"] = counters[metric][
                                                              "kth_valid"] / n_samples if n_samples > 0 else 0
                # AccTopK: 前k个中至少有一个正确的样本比例
                results[metric]["AccTopK"][f"K={k}"] = counters[metric][
                                                           "topk_correct"] / n_samples if n_samples > 0 else 0
                # AccKth: 第k个正确的样本比例
                results[metric]["AccKth"][f"K={k}"] = counters[metric][
                                                          "kth_correct"] / n_samples if n_samples > 0 else 0

    return results


def load_prediction_file(file_path: str) -> tuple:
    """
    加载预测文件并提取任务信息
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not data:
            raise ValueError("Prediction file is empty")

        # 从第一个样本的instruction中提取任务ID
        first_sample = data[0]
        instruction = first_sample.get("instruction", "")

        # 提取任务ID（去掉冒号）
        task_tag = instruction.strip()
        if task_tag not in TASK_TAG_DICT:
            raise ValueError(f"Unknown task ID: {task_tag}")

        # 从输出中获取num_return_sequences
        num_return_sequences = len(first_sample.get("output", []))

        return data, TASK_TAG_DICT[task_tag], num_return_sequences

    except Exception as e:
        logging.error(f"Error loading prediction file {file_path}: {str(e)}")
        raise


def main(args):
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 构建完整的文件路径
    prediction_file_path = os.path.join(args.prediction_dir, args.prediction_file)
    if not prediction_file_path.endswith('.json'):
        prediction_file_path += '.json'

    logging.info(f"Loading prediction file: {prediction_file_path}")

    try:
        # 加载数据
        data, task_id, num_return_sequences = load_prediction_file(prediction_file_path)
        logging.info(f"Loaded {len(data)} samples for task {task_id} with {num_return_sequences} return sequences")

        # 计算指标
        logging.info("Calculating metrics...")
        results = calculate_metrics(data, task_id, num_return_sequences)
        results["file"] = prediction_file_path

        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        # 构建输出文件路径
        output_file_name = args.prediction_file
        if output_file_name.endswith('.json'):
            output_file_name = output_file_name[:-5]
        output_file_path = os.path.join(args.output_dir, f"{output_file_name}_scores.json")

        # 保存结果
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Metrics calculated and saved to: {output_file_path}")

        # 打印主要结果
        print("\n" + "=" * 50)
        print(f"Evaluation Results for {task_id}")
        print("=" * 50)
        print(f"Total samples: {results['count']}")
        print(f"Return sequences: {results['num_return_sequences']}")

        for metric in TASKS[task_id]["metrics"]:
            print(f"\n--- {metric.upper()} ---")
            topk_key = 'TopK' if metric in ["totally", "resolved", "matched"] else "AccTopK"
            acc_top1 = results[metric][topk_key].get("K=1", 0)
            acc_topk = results[metric][topk_key].get(f"K={results['num_return_sequences']}", 0)
            print(f"AccTop1: {acc_top1:.6f}")
            print(f"AccTop{results['num_return_sequences']}: {acc_topk:.6f}")

            if metric == "upd_cano_am_prds":
                val_top1 = results[metric]["ValTopK"].get("K=1", 0)
                val_topk = results[metric]["ValTopK"].get(f"K={results['num_return_sequences']}", 0)
                print(f"ValTop1: {val_top1:.6f}")
                print(f"ValTop{results['num_return_sequences']}: {val_topk:.6f}")

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chemical Mechanism Prediction Scoring")

    # 文件路径参数
    parser.add_argument("--prediction_dir", type=str,
                        default="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/chemechpred/prediction/random100/updcanoamrxts_cls_to_mech_updcanoamprds")
    parser.add_argument("--prediction_file", type=str,
                        default="updcanoamrxts_cls_to_mech_updcanoamprds_para02",
                        help="Name of the prediction file (without .json extension)")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/chemechpred/scores/random100/updcanoamrxts_cls_to_mech_updcanoamprds")

    args = parser.parse_args()
    main(args)
