# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: task_conversion.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/20 17:06
# https://chat.deepseek.com/a/chat/s/dcdb0ad3-05e9-432d-8158-24965e5fd8d4

import json
import os
from pathlib import Path
from tqdm import tqdm


def process_chemical_mechanism_data(data_dir_pattern, output_dir_pattern, task_id, split):
    """
    处理化学机理数据并生成指定任务的JSON文件

    Args:
        data_dir_pattern: 输入数据目录模式（包含{split}占位符）
        output_dir_pattern: 输出数据目录模式（包含{split}占位符）
        task_id: 任务标识符
        split: 数据集分割（train/val/test）
    """

    # 格式化目录路径（替换 {split} 占位符）
    data_dir = data_dir_pattern
    output_dir = output_dir_pattern.format(split=split)

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 构建输入文件路径
    input_file = os.path.join(data_dir, f"mech-USPTO-31k_{split}.json")

    # 构建输出文件路径
    output_file = os.path.join(output_dir, f"{task_id.lower()}_{split}.json")

    print(f"处理任务: {task_id}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")

    # 读取输入数据
    if not os.path.exists(input_file):
        print(f"警告: 输入文件不存在 {input_file}，跳过该任务")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"读取到 {len(data)} 条数据")

    # 处理数据
    processed_data = []

    for item in tqdm(data, desc=f"处理 {split} 数据"):
        # 提取基本信息
        item_id = item["id"]
        reactants = item["updated_reaction"].split('>>')[0]
        products = item["updated_reaction"].split('>>')[1]
        mechanism_class = item["mechanistic_class"]
        mechanism_label = item["mechanistic_label"]

        # 构建任务特定的ID
        task_specific_id = f"{task_id.lower()}_{split}_{item_id}"

        # 根据任务标识符构建输入和输出
        input_parts = []
        output_parts = []

        # 解析任务标识符
        input_spec, output_spec = task_id.split('_TO_')

        # 处理输入部分
        if input_spec == "AMRXTS":
            input_parts.extend(["AMRTS", reactants])
        elif input_spec == "AMRXTS_MECH":
            input_parts.extend(["AMRTS", reactants, "MECH", mechanism_label])
        elif input_spec == "AMRXTS_CLS":
            input_parts.extend(["AMRTS", reactants, "CLS", mechanism_class])
        elif input_spec == "AMRXTS_MECH_CLS":
            input_parts.extend(["AMRTS", reactants, "MECH", mechanism_label, "CLS", mechanism_class])

        # 处理输出部分
        if output_spec == "AMPRDS":
            output_parts.extend(["AMPRDS", products])
        elif output_spec == "MECH_AMPRDS":
            output_parts.extend(["MECH", mechanism_label, "AMPRDS", products])
        elif output_spec == "CLS_AMPRDS":
            output_parts.extend(["CLS", mechanism_class, "AMPRDS", products])
        elif output_spec == "CLS_MECH_AMPRDS":
            output_parts.extend(["CLS", mechanism_class, "MECH", mechanism_label, "AMPRDS", products])

        # 构建最终的数据项
        processed_item = {
            "id": task_specific_id,
            "instruction": f"{task_id}:",
            "input": "\n".join(input_parts),
            "output": "\n".join(output_parts)
        }

        processed_data.append(processed_item)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"成功处理并保存 {len(processed_data)} 条数据到 {output_file}")
    print("-" * 50)


def main():
    # 配置参数
    data_dir_pattern = "/mnt/e/DataSets/Chemistry/ChemicalMechanism/processed"
    output_dir_pattern = "/mnt/e/DataSets/Chemistry/ChemicalMechanism/{split}"

    # 所有任务列表
    tasks = [
        "AMRXTS_TO_AMPRDS",
        "AMRXTS_TO_MECH_AMPRDS",
        "AMRXTS_TO_CLS_AMPRDS",
        "AMRXTS_TO_CLS_MECH_AMPRDS",
        "AMRXTS_MECH_TO_AMPRDS",
        "AMRXTS_CLS_TO_AMPRDS",
        "AMRXTS_MECH_CLS_TO_AMPRDS"
    ]

    # 所有数据集分割
    splits = ["train", "val", "test"]

    print("开始处理所有化学机理数据任务...")
    print("=" * 60)

    # 遍历所有任务和分割
    for task_id in tasks:
        for split in splits:
            try:
                process_chemical_mechanism_data(data_dir_pattern, output_dir_pattern, task_id, split)
            except Exception as e:
                print(f"处理任务 {task_id} 分割 {split} 时出错: {e}")
                continue

    print("所有任务处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()