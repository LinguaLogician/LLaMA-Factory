# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: score2.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/28 2:14

# https://chat.deepseek.com/a/chat/s/2230b7ad-64ef-449d-a448-338ee9ceef94

# -*- coding: utf-8 -*-
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import logging
from datetime import datetime

# ============================ 配置常量 ============================

DEFAULT_MAX_K = 10  # 默认计算前5个结果

# ============================ 评分计算器类 ============================
class ChemMechScoreCalculator:
    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.setup_logging()
        self.max_k = min(self.args.max_k, DEFAULT_MAX_K)

    def setup_paths(self):
        """设置文件路径"""
        # 解析group和task_id
        self.group = self.args.group
        self.task_id = self.args.task_id

        # 设置预测文件路径
        prediction_dir = Path(self.args.prediction_base_dir) / self.args.subset / self.group /self.task_id
        self.prediction_file = prediction_dir / f"{self.args.model_name}.json"

        if not self.prediction_file.exists():
            raise FileNotFoundError(f"预测文件不存在: {self.prediction_file}")

        # 设置输出路径
        self.output_dir = Path(self.args.output_base_dir) / self.args.subset / self.group / self.task_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.args.model_name}.json"

    def setup_logging(self):
        """设置日志"""
        log_dir = Path("./logs/chemechpred/scoring")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.args.model_name}_{self.group}_{self.task_id}_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_output_components(self, instruction: str) -> List[str]:
        """从instruction中提取输出组件名称"""
        output_part = instruction.split('->')[1]
        components = [comp.strip().lower().replace('.', '_') for comp in output_part.split('+')]
        return components

    def calculate_metrics_for_component(self, data: List[Dict], component: str, max_k: int) -> Dict[str, Any]:
        """计算单个组件的各项指标"""
        total_samples = len(data)
        metrics = {
            "AccTopK": {},
            "AccKth": {},
            "AccTopKMols": {},
            "AccKthMols": {},
            "PrecTopKFormat": {},
            "PrecKthFormat": {},
            "PrecTopKManner": {},
            "PrecKthManner": {},
            "PrecTopKValid": {},
            "PrecKthValid": {},
            "PrecTopK": {},  # 对应is_correct
            "PrecKth": {}  # 对应is_correct
        }

        # 初始化指标字典
        for k in range(1, max_k + 1):
            for metric in metrics.values():
                metric[f"K={k}"] = 0.0

        # 计算每个样本的指标
        for sample in tqdm(data, desc=f"计算组件 {component} 指标"):
            outputs = sample.get("output", [])
            if not outputs:
                continue

            # 限制到max_k个结果
            outputs = outputs[:max_k]

            # 提取该组件的评估结果
            component_results = []
            for output in outputs:
                comp_eval = output.get(component, {})
                component_results.append({
                    "is_correct": comp_eval.get("is_correct", False),
                    "is_correct_mols": comp_eval.get("is_correct_mols", False),
                    "is_correct_format": comp_eval.get("is_correct_format", False),
                    "is_correct_manner": comp_eval.get("is_correct_manner", False),
                    "is_valid": comp_eval.get("is_valid", False)
                })

            # 计算AccTopK和AccKth相关指标
            for k in range(1, max_k + 1):
                k_results = component_results[:k]

                # AccTopK: 前k个中至少有一个正确
                if any(result["is_correct"] for result in k_results):
                    metrics["AccTopK"][f"K={k}"] += 1
                if any(result["is_correct_mols"] for result in k_results):
                    metrics["AccTopKMols"][f"K={k}"] += 1

                # AccKth: 第k个正确（如果存在）
                if k <= len(k_results):
                    if k_results[k - 1]["is_correct"]:
                        metrics["AccKth"][f"K={k}"] += 1
                    if k_results[k - 1]["is_correct_mols"]:
                        metrics["AccKthMols"][f"K={k}"] += 1

            # 计算Precision相关指标（基于所有结果）
            total_results = len(outputs)
            for k in range(1, max_k + 1):
                k_results = component_results[:k]

                # PrecTopK: 前k个中正确结果的比例
                correct_count = sum(1 for result in k_results if result["is_correct"])
                metrics["PrecTopK"][f"K={k}"] += correct_count

                correct_mols_count = sum(1 for result in k_results if result["is_correct_mols"])
                # AccTopKMols已经计算过，这里计算Prec版本（如果需要）

                format_correct_count = sum(1 for result in k_results if result["is_correct_format"])
                metrics["PrecTopKFormat"][f"K={k}"] += format_correct_count

                manner_correct_count = sum(1 for result in k_results if result["is_correct_manner"])
                metrics["PrecTopKManner"][f"K={k}"] += manner_correct_count

                valid_count = sum(1 for result in k_results if result["is_valid"])
                metrics["PrecTopKValid"][f"K={k}"] += valid_count

                # PrecKth: 第k个结果的指标（如果存在）
                if k <= len(k_results):
                    kth_result = k_results[k - 1]
                    metrics["PrecKth"][f"K={k}"] += 1 if kth_result["is_correct"] else 0
                    metrics["PrecKthFormat"][f"K={k}"] += 1 if kth_result["is_correct_format"] else 0
                    metrics["PrecKthManner"][f"K={k}"] += 1 if kth_result["is_correct_manner"] else 0
                    metrics["PrecKthValid"][f"K={k}"] += 1 if kth_result["is_valid"] else 0

        # 转换为比率
        for metric_name, metric_dict in metrics.items():
            for k in range(1, max_k + 1):
                key = f"K={k}"
                if metric_name.startswith('Acc'):
                    # Accuracy指标除以样本总数
                    metric_dict[key] = metric_dict[key] / total_samples if total_samples > 0 else 0.0
                elif metric_name.startswith('PrecTopK'):
                    # Precision指标除以总结果数（样本数 × k）
                    total_possible = total_samples * k
                    metric_dict[key] = metric_dict[key] / total_possible if total_possible > 0 else 0.0
                elif metric_name.startswith('PrecKth'):
                    # Precision指标除以总结果数（样本数 × k）
                    total_possible = total_samples
                    metric_dict[key] = metric_dict[key] / total_possible if total_possible > 0 else 0.0

        return metrics

    def calculate_global_metrics(self, data: List[Dict], max_k: int) -> Dict[str, Any]:
        """计算全局指标（totally, resolved, matched）"""
        total_samples = len(data)
        global_metrics = {
            "totally": {"PrecTopK": {}, "PrecKth": {}},
            "resolved": {"PrecTopK": {}, "PrecKth": {}},
            "matched": {"PrecTopK": {}, "PrecKth": {}}
        }

        # 初始化指标字典
        for category in global_metrics:
            for metric_type in global_metrics[category]:
                for k in range(1, max_k + 1):
                    global_metrics[category][metric_type][f"K={k}"] = 0.0

        for sample in tqdm(data, desc="计算全局指标"):
            outputs = sample.get("output", [])
            if not outputs:
                continue

            outputs = outputs[:max_k]

            for k in range(1, max_k + 1):
                k_results = outputs[:k]

                # PrecTopK指标
                totally_correct_count = sum(1 for output in k_results if output.get("is_correct_totally", False))
                resolved_correct_count = sum(1 for output in k_results if output.get("is_resolved_correct", False))
                matched_correct_count = sum(1 for output in k_results if output.get("is_task_matched", False))

                global_metrics["totally"]["PrecTopK"][f"K={k}"] += totally_correct_count
                global_metrics["resolved"]["PrecTopK"][f"K={k}"] += resolved_correct_count
                global_metrics["matched"]["PrecTopK"][f"K={k}"] += matched_correct_count

                # PrecKth指标（第k个结果）
                if k <= len(k_results):
                    kth_output = k_results[k - 1]
                    global_metrics["totally"]["PrecKth"][f"K={k}"] += 1 if kth_output.get("is_correct_totally",
                                                                                          False) else 0
                    global_metrics["resolved"]["PrecKth"][f"K={k}"] += 1 if kth_output.get("is_resolved_correct",
                                                                                           False) else 0
                    global_metrics["matched"]["PrecKth"][f"K={k}"] += 1 if kth_output.get("is_task_matched",
                                                                                          False) else 0

        # 转换为比率
        for category in global_metrics:
            for metric_type in global_metrics[category]:
                for k in range(1, max_k + 1):
                    key = f"K={k}"
                    total_possible = total_samples
                    if metric_type=="PrecTopK":
                        total_possible = total_samples * k
                    elif metric_type == "PrecKth":
                        total_possible = total_samples
                    global_metrics[category][metric_type][key] = (
                        global_metrics[category][metric_type][key] / total_possible
                        if total_possible > 0 else 0.0
                    )


        return global_metrics

    def calculate_scores(self) -> Dict[str, Any]:
        """计算所有评分指标"""
        self.logger.info(f"开始计算评分: {self.prediction_file}")

        # 加载预测数据
        with open(self.prediction_file, 'r') as f:
            data = json.load(f)

        if not data:
            raise ValueError("预测数据为空")

        total_samples = len(data)
        self.logger.info(f"数据量: {total_samples}")

        # 提取任务信息
        first_sample = data[0]
        task_id = first_sample.get("instruction", "").replace('->', '_TO_').replace('.', '').upper()

        # 提取输出组件
        components = self.extract_output_components(first_sample["instruction"])
        self.logger.info(f"任务组件: {components}")

        # 计算各组件指标
        component_metrics = {}
        for component in components:
            self.logger.info(f"计算组件 {component} 的指标...")
            component_metrics[component] = self.calculate_metrics_for_component(data, component, self.max_k)

        # 计算全局指标
        self.logger.info("计算全局指标...")
        global_metrics = self.calculate_global_metrics(data, self.max_k)

        # 组装最终结果
        result = {
            "file": str(self.prediction_file),
            "task": task_id,
            "count": total_samples,
            "max_k": self.max_k,
            **component_metrics,
            **global_metrics
        }

        # 保存结果
        with open(self.output_file, 'w') as f:
            json.dump(result, f, indent=2)

        self.logger.info(f"评分计算完成，结果保存至: {self.output_file}")

        # 打印摘要信息
        self.print_summary(result)

        return result

    def print_summary(self, result: Dict[str, Any]):
        """打印结果摘要"""
        self.logger.info("=" * 50)
        self.logger.info("评分结果摘要")
        self.logger.info("=" * 50)
        self.logger.info(f"任务: {result['task']}")
        self.logger.info(f"样本数: {result['count']}")
        self.logger.info(f"最大K值: {result['max_k']}")

        # 打印各组件K=1的准确率
        for comp_name, metrics in result.items():
            if comp_name in ['file', 'task', 'count', 'max_k', 'totally', 'resolved', 'matched']:
                continue

            if 'AccTopK' in metrics and 'K=1' in metrics['AccTopK']:
                acc = metrics['AccTopK']['K=1']
                self.logger.info(f"{comp_name} - AccTop1: {acc:.4f}")

        # 打印全局指标
        if 'totally' in result and 'PrecTopK' in result['totally']:
            totally_prec = result['totally']['PrecTopK']['K=1']
            resolved_prec = result['resolved']['PrecTopK']['K=1']
            matched_prec = result['matched']['PrecTopK']['K=1']

            self.logger.info(
                f"全局指标 - Totally: {totally_prec:.4f}, Resolved: {resolved_prec:.4f}, Matched: {matched_prec:.4f}")


def main():
    parser = argparse.ArgumentParser(description="化学机制预测评分计算脚本")

    # 必需参数
    parser.add_argument("--group", type=str, default="prds_to_prds",
                        help="group")
    parser.add_argument("--task_id", type=str, default="updcanoamprds_to_updcanostdprds",
                        help="task_id")
    parser.add_argument("--model_name", type=str, default="updcanoamprds_to_updcanostdprds",
                        help="模型名称")
    # 路径参数
    parser.add_argument("--prediction_base_dir", type=str,
                        default="results/chemechpred/prediction",
                        help="预测结果基础目录")
    parser.add_argument("--output_base_dir", type=str,
                        default="results/chemechpred/scores",
                        help="评分结果输出基础目录")
    parser.add_argument("--subset", type=str, default="_random313")

    # 计算参数
    parser.add_argument("--max_k", type=int, default=5,
                        help="计算的最大K值")

    args = parser.parse_args()

    try:
        calculator = ChemMechScoreCalculator(args)
        result = calculator.calculate_scores()

        print(f"评分计算完成！结果保存至: {calculator.output_file}")

    except Exception as e:
        logging.error(f"评分计算失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()