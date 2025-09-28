# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_sampling.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/21 15:03
# https://chat.deepseek.com/a/chat/s/be44bd6a-1108-4677-98c0-6fc9a0de2295
# https://chat.deepseek.com/a/chat/s/03cb1520-2cdb-4125-b80d-d0857897d6f1
# https://chat.deepseek.com/a/chat/s/1abbd2ca-4a91-4705-b751-8421720fa9bf

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def sample_json_files(data_dir, sample_sizes, base_file="mech_USPTO.json"):
    """
    以mech_USPTO.json文件为基准，根据id的数字部分进行随机采样

    Args:
        data_dir: 数据目录路径
        sample_sizes: 采样大小的列表，如[100, 200, 500]
        base_file: 基准文件名
    """
    data_dir = Path(data_dir)

    # 步骤1: 读取基准文件，提取所有id的数字部分
    print(f"步骤1: 读取基准文件 {base_file}")
    base_file_path = data_dir / base_file

    if not base_file_path.exists():
        print(f"错误: 基准文件 {base_file_path} 不存在")
        return

    with open(base_file_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    # 提取基准文件中的所有数字id
    base_ids = set()
    for item in base_data:
        if 'id' in item:
            # 提取数字部分
            id_str = item['id']
            numbers = ''.join(filter(str.isdigit, id_str))
            if numbers:
                base_ids.add(numbers)

    print(f"从基准文件中提取到 {len(base_ids)} 个唯一数字ID")

    # 步骤2: 收集所有JSON文件
    print("步骤2: 收集所有JSON文件")
    json_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                full_path = Path(root) / file
                # 计算相对于data_dir的路径
                rel_path = full_path.relative_to(data_dir)
                json_files.append(rel_path)

    print(f"找到 {len(json_files)} 个JSON文件")

    # 步骤3: 为每个采样大小创建处理流程
    for sample_size in sample_sizes:
        print(f"\n步骤3: 处理采样大小 {sample_size}")

        # 从基准id中随机采样
        if sample_size > len(base_ids):
            print(f"警告: 采样大小 {sample_size} 大于可用ID数量 {len(base_ids)}，使用所有可用ID")
            sampled_ids = base_ids
        else:
            sampled_ids = set(random.sample(list(base_ids), sample_size))

        # 创建目标目录
        target_dir = data_dir / f"_random{sample_size}"
        target_dir.mkdir(exist_ok=True)
        print(f"创建目标目录: {target_dir}")

        # 步骤4: 处理每个JSON文件
        print("步骤4: 处理各个JSON文件")
        for json_file_rel in tqdm(json_files, desc=f"采样 {sample_size}"):
            json_file_path = data_dir / json_file_rel
            target_file_path = target_dir / json_file_rel

            # 确保目标目录存在
            target_file_path.parent.mkdir(parents=True, exist_ok=True)

            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                try:
                    file_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"警告: 文件 {json_file_path} JSON解析错误: {e}")
                    continue

            # 筛选匹配采样id的数据
            filtered_data = []
            for item in file_data:
                if 'id' in item:
                    item_id = item['id'].split('_')[-1]
                    numbers = ''.join(filter(str.isdigit, item_id))
                    if numbers in sampled_ids:
                        filtered_data.append(item)

            # 写入目标文件
            with open(target_file_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        print(f"采样 {sample_size} 完成，文件保存在 {target_dir}")


def main():
    parser = argparse.ArgumentParser(description='根据基准文件随机采样JSON数据')
    parser.add_argument('--data_dir', type=str, default='/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/test',
                        help='数据目录路径')
    parser.add_argument('--sample_sizes', type=int, nargs='+', default=[313],
                        help='采样大小列表')
    parser.add_argument('--base_file', type=str, default='mech_USPTO.json',
                        help='基准文件名')

    args = parser.parse_args()

    print("开始执行JSON文件采样程序")
    print(f"数据目录: {args.data_dir}")
    print(f"采样大小: {args.sample_sizes}")
    print(f"基准文件: {args.base_file}")
    print("-" * 50)

    sample_json_files(args.data_dir, args.sample_sizes, args.base_file)

    print("\n程序执行完成！")


if __name__ == "__main__":
    main()