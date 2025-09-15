# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: generate_hardxmp.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/15 9:04
# https://chat.deepseek.com/a/chat/s/e15955c7-91c6-46a4-a87e-67eb8ca056f7
import json
import os
import argparse
import logging
import math
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path


def setup_logging():
    """设置日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def calculate_hardness_score(outputs: List[Dict]) -> float:
    """
    计算样本的困难程度

    参数:
        outputs: 输出结果列表

    返回:
        困难程度分数
    """
    total_score = sum(item["sequence_score"] for item in outputs)
    correct_scores_sum = sum(item["sequence_score"] for item in outputs if item.get("is_correct", False))

    # 计算分数比值，避免除以零
    if total_score < 1e-5:
        ratio = 1
    else:
        ratio = total_score / (correct_scores_sum + 1e-5)

    # 取log10并除以5
    hardness = math.log10(ratio) / 5

    return hardness


def process_data(data: List[Dict]) -> List[Dict]:
    """
    处理数据，为每条数据计算hardness分数

    参数:
        data: 原始数据列表

    返回:
        处理后的数据列表
    """
    processed_data = []

    for item in tqdm(data, desc="计算困难分数"):
        # 复制原始数据，避免修改原数据
        processed_item = item.copy()

        # 计算hardness分数
        hardness = calculate_hardness_score(item["output"])
        processed_item["hardness"] = hardness

        # 将hardness插入到prompt_length和output之间
        # 首先移除output
        output = processed_item.pop("output")

        # 重新构建字典，确保顺序正确
        new_item = {}
        for key in processed_item.keys():
            new_item[key] = processed_item[key]
            if key == "prompt_length":
                new_item["hardness"] = hardness

        new_item["output"] = output
        processed_data.append(new_item)

    return processed_data


def sort_data_by_hardness(data: List[Dict]) -> List[Dict]:
    """
    根据hardness对数据进行排序（从小到大）

    参数:
        data: 数据列表

    返回:
        排序后的数据列表
    """
    return sorted(data, key=lambda x: x["hardness"])


def split_data_by_hardness(data: List[Dict], thresholds: List[float]) -> List[List[Dict]]:
    """
    根据hardness阈值划分数据

    参数:
        data: 处理后的数据列表
        thresholds: 阈值列表

    返回:
        划分后的数据列表
    """
    # 初始化划分结果
    splits = []

    # 添加第一个区间 (0, thresholds[0]]
    splits.append([item for item in data if item["hardness"] <= thresholds[0]])

    # 添加中间区间 (thresholds[i], thresholds[i+1]]
    for i in range(len(thresholds) - 1):
        splits.append([
            item for item in data
            if thresholds[i] < item["hardness"] <= thresholds[i + 1]
        ])

    # 添加最后一个区间 (thresholds[-1], 1]
    splits.append([item for item in data if item["hardness"] > thresholds[-1]])

    return splits


def save_json(data: List[Dict], filepath: str):
    """保存数据到JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_summary(data: List[Dict], splits: List[List[Dict]], thresholds: List[float],
                     base_name: str, target_dir: str):
    """
    生成数据集摘要信息

    参数:
        data: 所有数据
        splits: 划分后的数据
        thresholds: 阈值列表
        base_name: 基础文件名
        target_dir: 目标目录
    """
    summary_path = os.path.join(target_dir, f"{base_name}_summary.txt")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"数据集摘要: {base_name}\n")
        f.write("=" * 50 + "\n\n")

        # 总体信息
        f.write(f"总数据量: {len(data)}\n\n")

        # 各划分信息
        ranges = []
        ranges.append(f"(0, {thresholds[0]}]")

        for i in range(len(thresholds) - 1):
            ranges.append(f"({thresholds[i]}, {thresholds[i + 1]}]")

        ranges.append(f"({thresholds[-1]}, 1]")

        for i, split_data in enumerate(splits):
            count = len(split_data)
            percentage = (count / len(data)) * 100 if len(data) > 0 else 0

            f.write(f"划分 {i + 1} ({ranges[i]}):\n")
            f.write(f"  数据量: {count}\n")
            f.write(f"  占比: {percentage:.2f}%\n")

            # 计算该划分的平均hardness
            if count > 0:
                avg_hardness = sum(item["hardness"] for item in split_data) / count
                f.write(f"  平均困难度: {avg_hardness:.4f}\n")

            f.write("\n")

        # 整体统计信息
        if data:
            min_hardness = min(item["hardness"] for item in data)
            max_hardness = max(item["hardness"] for item in data)
            avg_hardness = sum(item["hardness"] for item in data) / len(data)

            f.write("整体统计:\n")
            f.write(f"  最小困难度: {min_hardness:.4f}\n")
            f.write(f"  最大困难度: {max_hardness:.4f}\n")
            f.write(f"  平均困难度: {avg_hardness:.4f}\n")


def main(data_dir: str, data_file: str, target_dir: str, hardness_thresholds: List[float]):
    """
    主处理函数

    参数:
        data_dir: 数据目录
        data_file: 数据文件名
        target_dir: 目标目录
        hardness_thresholds: 困难度阈值列表
    """
    # 设置日志
    setup_logging()

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 构建完整文件路径
    data_path = os.path.join(data_dir, data_file)

    # 读取数据
    logging.info(f"正在读取数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理数据，计算hardness分数
    processed_data = process_data(data)

    # 对完整数据进行排序
    sorted_data = sort_data_by_hardness(processed_data)

    # 提取基础文件名（不含扩展名）
    base_name = os.path.splitext(data_file)[0]

    # 保存完整数据（包含output）
    all_data_path = os.path.join(target_dir, f"{base_name}_all.json")
    logging.info(f"正在保存完整数据: {all_data_path}")
    save_json(sorted_data, all_data_path)

    # 创建精简数据（不包含output）
    tidy_data = []
    for item in sorted_data:
        tidy_item = item.copy()
        tidy_item.pop("output", None)
        tidy_data.append(tidy_item)

    # 对精简数据进行排序（虽然已经是排序的，但为了确保）
    tidy_data = sort_data_by_hardness(tidy_data)

    # 保存精简数据
    tidy_data_path = os.path.join(target_dir, f"{base_name}_tidy.json")
    logging.info(f"正在保存精简数据: {tidy_data_path}")
    save_json(tidy_data, tidy_data_path)

    # 根据hardness划分数据
    logging.info("正在根据困难度划分数据")
    splits = split_data_by_hardness(sorted_data, hardness_thresholds)

    # 保存划分后的完整数据
    for i, split_data in enumerate(splits, 1):
        # 确保每个划分内的数据也是排序的
        split_data_sorted = sort_data_by_hardness(split_data)
        split_path = os.path.join(target_dir, f"{base_name}_h{i}.json")
        logging.info(f"正在保存划分 {i} 的完整数据: {split_path}")
        save_json(split_data_sorted, split_path)

    # 根据hardness划分精简数据
    tidy_splits = split_data_by_hardness(tidy_data, hardness_thresholds)

    # 保存划分后的精简数据
    for i, split_data in enumerate(tidy_splits, 1):
        # 确保每个划分内的数据也是排序的
        split_data_sorted = sort_data_by_hardness(split_data)
        split_path = os.path.join(target_dir, f"{base_name}_ht{i}.json")
        logging.info(f"正在保存划分 {i} 的精简数据: {split_path}")
        save_json(split_data_sorted, split_path)

    # 生成摘要
    logging.info("正在生成摘要")
    generate_summary(sorted_data, splits, hardness_thresholds, base_name, target_dir)

    logging.info("处理完成!")


if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="处理数据并计算困难度分数")

    # 添加参数
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/prediction/retrosyn_nospace_test",
                        help="数据目录路径")
    parser.add_argument("--data_file", type=str,
                        default="qwen205_retrosyn_nospace_full_para1_ckptlast.json",
                        help="数据文件名")
    parser.add_argument("--target_dir", type=str,
                        default="/mnt/e/DataSets/Chemistry/RetroPrediction/HardExamples/prediction_test/qwen205_retrosyn_nospace_full_para1_ckptlast",
                        help="目标目录路径")
    parser.add_argument("--hardness_thresholds", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6, 0.8],
                        help="困难度阈值列表")

    # 解析参数
    args = parser.parse_args()

    # 调用主函数
    main(args.data_dir, args.data_file, args.target_dir, args.hardness_thresholds)