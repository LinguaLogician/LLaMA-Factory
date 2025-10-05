# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_conversion.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/19 22:16
# https://chat.deepseek.com/a/chat/s/14fa2cfa-68cd-48bc-bd3d-ea0be451a05f

import os
import csv
import json
import argparse
from tqdm import tqdm


def convert_csv_to_json(data_dir="/mnt/e/DataSets/Chemistry/ChemicalMechanism",
                        data_file="mech-USPTO-31k.csv",
                        output_dir="processed"):
    """
    将CSV文件转换为JSON格式

    Args:
        data_dir: 数据目录路径
        data_file: CSV文件名
        output_dir: 输出目录名称
    """
    # 构建完整文件路径
    csv_file_path = os.path.join(data_dir, data_file)
    output_dir_path = os.path.join(data_dir, output_dir)
    json_file_name = os.path.splitext(data_file)[0] + ".json"
    json_file_path = os.path.join(output_dir_path, json_file_name)

    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: CSV文件不存在: {csv_file_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"创建输出目录: {output_dir_path}")

    # 读取CSV文件
    print(f"开始读取CSV文件: {csv_file_path}")
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            rows = list(csv_reader)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    print(f"成功读取 {len(rows)} 条数据")

    # 处理数据
    print("开始处理数据...")
    json_data = []

    for i, row in enumerate(tqdm(rows, desc="处理进度")):
        # 生成ID
        item_id = f"chemical_mechanism_{i:08d}"

        # 构建JSON对象
        json_item = {
            "id": item_id,
            "original_reactions": row.get("original_reactions", ""),
            "updated_reaction": row.get("updated_reaction", ""),
            "mechanistic_class": row.get("mechanistic_class", ""),
            "mechanistic_label": row.get("mechanistic_label", "")
        }

        json_data.append(json_item)

    # 保存为JSON文件
    print(f"开始保存JSON文件: {json_file_path}")
    try:
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)
        print(f"成功保存JSON文件，共 {len(json_data)} 条数据")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将CSV文件转换为JSON格式")
    parser.add_argument("--data_dir", default="/mnt/e/DataSets/Chemistry/ChemicalMechanism",
                        help="数据目录路径")
    parser.add_argument("--data_file", default="mech-USPTO-31k.csv",
                        help="CSV文件名")
    parser.add_argument("--output_dir", default="processed",
                        help="输出目录名称")

    args = parser.parse_args()

    print("=" * 50)
    print("CSV转JSON工具")
    print("=" * 50)
    print(f"数据目录: {args.data_dir}")
    print(f"输入文件: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 50)

    convert_csv_to_json(args.data_dir, args.data_file, args.output_dir)

    print("=" * 50)
    print("处理完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
