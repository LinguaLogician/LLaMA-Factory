# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_sampling.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/21 15:03
# https://chat.deepseek.com/a/chat/s/be44bd6a-1108-4677-98c0-6fc9a0de2295
import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def process_json_files(data_dir, sample_sizes, random_seed=42):
    """
    处理JSON文件并进行随机采样

    Args:
        data_dir: 数据目录路径
        sample_sizes: 采样大小列表，如 [100, 200, 500]
        random_seed: 随机种子，确保结果可重现
    """
    # 设置随机种子以确保结果可重现
    random.seed(random_seed)

    # 确保数据目录存在
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"错误：数据目录 '{data_dir}' 不存在")
        return

    # 获取所有JSON文件
    json_files = list(data_path.glob("*.json"))
    if not json_files:
        print(f"在目录 '{data_dir}' 中未找到JSON文件")
        return

    print(f"找到 {len(json_files)} 个JSON文件")

    # 为每个采样大小创建输出目录
    output_dirs = {}
    for size in sample_sizes:
        output_dir = data_path / f"random{size}"
        output_dir.mkdir(exist_ok=True)
        output_dirs[size] = output_dir
        print(f"创建输出目录: {output_dir}")

    # 处理每个文件
    for file_path in tqdm(json_files, desc="处理文件"):
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查数据是否为列表
            if not isinstance(data, list):
                print(f"警告：文件 {file_path.name} 不包含JSON数组，跳过")
                continue

            total_items = len(data)

            # 对每个采样大小进行处理
            for size in sample_sizes:
                if total_items <= size:
                    # 如果数据量小于等于采样大小，直接复制文件
                    output_path = output_dirs[size] / file_path.name
                    shutil.copy2(file_path, output_path)
                    tqdm.write(f"文件 {file_path.name} 数据量({total_items}) <= {size}，直接复制")
                else:
                    # 随机采样
                    sampled_data = random.sample(data, size)

                    # 保存采样结果
                    output_path = output_dirs[size] / file_path.name
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(sampled_data, f, ensure_ascii=False, indent=2)

                    tqdm.write(f"文件 {file_path.name} 采样 {size} 条数据完成")

        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {str(e)}")

    print("所有文件处理完成！")


def main():
    """主函数，处理命令行参数并执行采样操作"""
    parser = argparse.ArgumentParser(description="JSON文件随机采样工具")
    parser.add_argument("--data_dir", type=str, default="/mnt/e/DataSets/Chemistry/ChemicalMechanism/test",
                        help="数据目录路径，包含JSON文件")
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=[100, 200, 500],
                        help="采样大小列表，用空格分隔")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子，确保结果可重现")

    args = parser.parse_args()

    print(f"开始处理目录: {args.data_dir}")
    print(f"采样大小: {args.sample_sizes}")
    print(f"随机种子: {args.random_seed}")
    print("-" * 50)

    process_json_files(args.data_dir, args.sample_sizes, args.random_seed)


if __name__ == "__main__":
    main()
