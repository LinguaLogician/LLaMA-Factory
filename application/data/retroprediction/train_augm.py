# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: train_augm.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/15 13:35
# https://chat.deepseek.com/a/chat/s/84ad568e-bcc9-4ba5-a299-8b815c14eace
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from rdkit import Chem
import random


def canonicalize_smiles(smiles):
    """将SMILES字符串转换为规范形式"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return smiles
    except:
        return smiles


def randomize_smiles(smiles, num_variants=1):
    """生成SMILES的随机变体"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles] * num_variants

        results = []
        for _ in range(num_variants):
            random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            results.append(random_smiles)

        return results
    except:
        return [smiles] * num_variants


def process_data(data_dir, data_file, output_dir, version, num_augmentations=1):
    """处理数据并生成增强版本"""
    # 创建输出目录
    output_path = os.path.join(output_dir, version)
    os.makedirs(output_path, exist_ok=True)

    # 读取原始数据
    input_file = os.path.join(data_dir, data_file)
    print(f"正在读取数据文件: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"成功读取 {len(data)} 条数据")

    # 处理output增强
    output_augm_data = []
    print("正在生成output增强数据...")
    for item in tqdm(data):
        original_output = item["output"]
        randomized_outputs = randomize_smiles(original_output, num_augmentations)

        for i, random_output in enumerate(randomized_outputs):
            new_item = item.copy()
            new_item["id"] = f"output_augm_{version}_{item['id']}_{i}"
            new_item["output"] = random_output
            output_augm_data.append(new_item)


    input_augm_data = []
    print("正在生成input增强数据...")
    for item in tqdm(data):
        original_input = item["input"]
        randomized_inputs = randomize_smiles(original_input, num_augmentations)

        for i, random_input in enumerate(randomized_inputs):
            new_item = item.copy()
            new_item["id"] = f"input_augm_{version}_{item['id']}_{i}"
            new_item["input"] = random_input
            input_augm_data.append(new_item)

    # 处理input和output增强
    inout_augm_data = []
    print("正在生成input和output增强数据...")
    for item in tqdm(data):
        original_input = item["input"]
        original_output = item["output"]

        randomized_inputs = randomize_smiles(original_input, num_augmentations)
        randomized_outputs = randomize_smiles(original_output, num_augmentations)

        for i, (random_input, random_output) in enumerate(zip(randomized_inputs, randomized_outputs)):
            new_item = item.copy()
            new_item["id"] = f"inout_augm_{version}_{item['id']}_{i}"
            new_item["input"] = random_input
            new_item["output"] = random_output
            inout_augm_data.append(new_item)

    # 保存增强后的数据
    data_file_name = data_file.split('.')[0]
    input_augm_file = os.path.join(output_path, f"{data_file_name}_input_augm.json")
    output_augm_file = os.path.join(output_path, f"{data_file_name}_output_augm.json")
    inout_augm_file = os.path.join(output_path, f"{data_file_name}_inout_augm.json")

    with open(output_augm_file, 'w') as f:
        json.dump(output_augm_data, f, indent=2)

    with open(inout_augm_file, 'w') as f:
        json.dump(inout_augm_data, f, indent=2)

    with open(input_augm_file, 'w') as f:
        json.dump(input_augm_data, f, indent=2)

    print(f"output增强数据已保存至: {output_augm_file}")
    print(f"input增强数据已保存至: {input_augm_file}")
    print(f"inout增强数据已保存至: {inout_augm_file}")

    return output_path, len(input_augm_data), len(output_augm_data), len(inout_augm_data)


def generate_summary(data_dir, data_file, output_dir, version,
                     original_count, input_augm_count, output_augm_count, inout_augm_count):
    """生成数据摘要信息"""
    data_file_name = data_file.split('.')[0]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "original_data": {
            "path": os.path.join(data_dir, data_file),
            "count": original_count
        },
        "input_augmentation": {
            "path": os.path.join(output_dir, version, f"{data_file_name}_input_augm.json"),
            "count": input_augm_count
        },
        "output_augmentation": {
            "path": os.path.join(output_dir, version, f"{data_file_name}_output_augm.json"),
            "count": output_augm_count
        },
        "inout_augmentation": {
            "path": os.path.join(output_dir, version, f"{data_file_name}_inout_augm.json"),
            "count": inout_augm_count
        },
        "augmentation_ratio": {
            "output": output_augm_count / original_count,
            "inout": inout_augm_count / original_count
        }
    }

    summary_file = os.path.join(output_dir, version, f"{data_file_name}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"摘要信息已保存至: {summary_file}")

    # 打印摘要信息
    print("\n=== 数据增强摘要 ===")
    print(f"处理时间: {summary['timestamp']}")
    print(f"原始数据条数: {original_count}")
    print(f"input增强后条数: {input_augm_count} (增强倍数: {summary['augmentation_ratio']['output']:.2f})")
    print(f"output增强后条数: {output_augm_count} (增强倍数: {summary['augmentation_ratio']['output']:.2f})")
    print(f"inout增强后条数: {inout_augm_count} (增强倍数: {summary['augmentation_ratio']['inout']:.2f})")

    return summary


def main():
    # 处理数据
    output_path, input_augm_count, output_augm_count, inout_augm_count = process_data(
        args.data_dir, args.data_file, args.output_dir, args.version, args.num_augmentations
    )

    # 读取原始数据以获取原始数量
    with open(os.path.join(args.data_dir, args.data_file), 'r') as f:
        original_data = json.load(f)

    # 生成摘要
    generate_summary(
        args.data_dir, args.data_file, args.output_dir, args.version,
        len(original_data), input_augm_count, output_augm_count, inout_augm_count
    )

    print("数据处理完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="化学数据增强工具")
    parser.add_argument("--data_dir", type=str, default="/mnt/e/DataSets/Chemistry/RetroPrediction/HardExamples/train/rank",
                        help="原始数据目录")
    parser.add_argument("--data_file", type=str, default="ht1.json",
                        help="原始数据文件名")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/e/DataSets/Chemistry/RetroPrediction/Augmented/train",
                        help="输出目录")
    parser.add_argument("--version", type=str, default="v4",
                        help="版本标识")
    parser.add_argument("--num_augmentations", type=int, default=1,
                        help="每条数据生成的增强版本数量")

    args = parser.parse_args()

    print("开始化学数据增强处理...")
    print(f"数据目录: {args.data_dir}")
    print(f"数据文件: {args.data_file}")
    print(f"输出目录: {args.output_dir}/{args.version}")
    print(f"增强倍数: {args.num_augmentations}")
    main()