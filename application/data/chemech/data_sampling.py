# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_sampling.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/21 15:03
# https://chat.deepseek.com/a/chat/s/be44bd6a-1108-4677-98c0-6fc9a0de2295
# https://chat.deepseek.com/a/chat/s/03cb1520-2cdb-4125-b80d-d0857897d6f1

import os
import json
import random
import argparse
from tqdm import tqdm
import shutil


def process_files(data_dir, base_filename, sample_sizes, random_seed=42):
    """
    根据基准文件随机采样其他文件，保持id数字部分一致

    Args:
        data_dir: 数据目录路径
        base_filename: 基准文件名
        sample_sizes: 采样大小列表，如[100, 200, 500]
        random_seed: 随机种子，确保可重复性
    """

    # 设置随机种子
    random.seed(random_seed)

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return

    # 获取所有JSON文件
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"找到 {len(all_files)} 个JSON文件")

    if base_filename not in all_files:
        print(f"错误: 基准文件 {base_filename} 不存在于数据目录中")
        return

    # 读取基准文件
    base_file_path = os.path.join(data_dir, base_filename)
    print(f"读取基准文件: {base_file_path}")

    with open(base_file_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    print(f"基准文件包含 {len(base_data)} 个元素")

    # 提取基准文件中的id数字部分
    base_ids = []
    for item in base_data:
        if 'id' in item:
            # 提取数字部分
            id_str = str(item['id'])
            # 查找最后一个下划线后的数字
            parts = id_str.split('_')
            if parts:
                numeric_part = parts[-1]
                if numeric_part.isdigit():
                    base_ids.append(numeric_part)

    print(f"从基准文件提取了 {len(base_ids)} 个有效ID")

    if not base_ids:
        print("错误: 基准文件中没有找到有效的ID")
        return

    # 为每个采样大小创建目录
    for size in sample_sizes:
        sample_dir = os.path.join(data_dir, f'random{size}')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            print(f"创建目录: {sample_dir}")

    # 对每个采样大小进行处理
    for sample_size in sample_sizes:
        print(f"\n开始处理采样大小: {sample_size}")

        # 如果采样大小大于基准数据量，使用全部数据
        actual_size = min(sample_size, len(base_ids))
        if actual_size < sample_size:
            print(f"警告: 采样大小 {sample_size} 大于基准数据量 {len(base_ids)}，使用全部数据")

        # 随机采样ID
        sampled_ids = random.sample(base_ids, actual_size)
        sampled_ids_set = set(sampled_ids)
        print(f"随机采样了 {len(sampled_ids)} 个ID")

        # 处理每个文件
        for filename in tqdm(all_files, desc=f"处理文件 (size={sample_size})"):
            file_path = os.path.join(data_dir, filename)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)

            # 筛选匹配的数据
            filtered_data = []
            for item in file_data:
                if 'id' in item:
                    id_str = str(item['id'])
                    parts = id_str.split('_')
                    if parts:
                        numeric_part = parts[-1]
                        if numeric_part in sampled_ids_set:
                            filtered_data.append(item)

            # 保存到目标目录
            target_dir = os.path.join(data_dir, f'random{sample_size}')
            target_path = os.path.join(target_dir, filename)

            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        print(f"采样大小 {sample_size} 处理完成，文件保存到 {os.path.join(data_dir, f'random{sample_size}')}")


def main():
    """主函数，处理命令行参数并执行处理"""
    parser = argparse.ArgumentParser(description='根据基准文件随机采样JSON数据')

    # 定义命令行参数，设置默认值
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/test',
                        help='数据目录路径')
    parser.add_argument('--base_file', type=str,
                        default='mech_USPTO.json',
                        help='基准文件名')
    parser.add_argument('--sample_sizes', type=int, nargs='+',
                        default=[100, 200, 500],
                        help='采样大小列表')
    parser.add_argument('--random_seed', type=int,
                        default=42,
                        help='随机种子')

    args = parser.parse_args()

    print("开始处理文件...")
    print(f"数据目录: {args.data_dir}")
    print(f"基准文件: {args.base_file}")
    print(f"采样大小: {args.sample_sizes}")
    print(f"随机种子: {args.random_seed}")
    print("-" * 50)

    # 执行处理
    process_files(args.data_dir, args.base_file, args.sample_sizes, args.random_seed)

    print("\n处理完成！")


if __name__ == "__main__":
    main()