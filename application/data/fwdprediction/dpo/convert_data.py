# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: convert_data.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/14 13:13
# https://chat.deepseek.com/a/chat/s/70969a14-62e5-4923-bcff-21cf92d7097c

import json
import os
import random
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_data(data_dir, data_file, target_dir, target_file):
    """
    将源数据文件转换为目标格式的DPO训练数据

    Args:
        data_dir: 源数据目录
        data_file: 源数据文件名
        target_dir: 目标目录
        target_file: 目标文件名
    """

    # 构建完整路径
    source_path = os.path.join(data_dir, data_file)
    target_path = os.path.join(target_dir, target_file)

    # 检查源文件是否存在
    if not os.path.exists(source_path):
        print(f"错误: 源文件 {source_path} 不存在!")
        return False

    print(f"正在读取源文件: {source_path}")

    try:
        # 读取源数据
        with open(source_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except Exception as e:
        print(f"读取源文件时出错: {e}")
        return False

    print(f"成功读取 {len(source_data)} 条数据")

    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    print(f"目标目录已创建/确认: {target_dir}")

    # 转换数据
    converted_data = []
    skipped_count = 0

    print("开始转换数据...")
    for item in tqdm(source_data, desc="处理进度"):
        # 获取无效输出（is_correct=false）
        rejected_outputs = [output for output in item.get('output', [])
                            if not output.get('is_correct', True)]

        # 如果没有无效输出，跳过此项
        if not rejected_outputs:
            skipped_count += 1
            continue

        # 随机选择一个无效输出
        rejected_output = random.choice(rejected_outputs)

        # 构建转换后的数据项
        converted_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"{item.get('instruction', '')}\n{item.get('input', '')}"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": item.get('label', '')
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_output.get('text', '')
            }
        }

        converted_data.append(converted_item)

    print(f"转换完成: 成功转换 {len(converted_data)} 条数据，跳过 {skipped_count} 条数据")

    # 保存转换后的数据
    try:
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        print(f"数据已保存到: {target_path}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False


def main():

    print("=" * 50)
    print("数据转换工具 - DPO格式转换")
    print("=" * 50)
    print(f"源目录: {args.data_dir}")
    print(f"源文件: {args.data_file}")
    print(f"目标目录: {args.target_dir}")
    print(f"目标文件: {args.target_file}")
    print("=" * 50)

    success = convert_data(
        data_dir=args.data_dir,
        data_file=args.data_file,
        target_dir=args.target_dir,
        target_file=args.target_file
    )

    if success:
        print("转换成功完成!")
    else:
        print("转换过程中出现错误!")

    print("=" * 50)


if __name__ == "__main__":
    """主函数，处理命令行参数并执行转换"""
    parser = argparse.ArgumentParser(description='将数据转换为DPO训练格式')
    parser.add_argument('--data_dir', type=str,
                        default='/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/valid/mit_mixed_augm_nospace_valid/',
                        help='源数据目录路径')
    parser.add_argument('--data_file', type=str,
                        default='qwen205_moltrans_mit_mixed_augm_rlhf_sft_lora_para1_ckptlast.json',
                        help='源数据文件名')
    parser.add_argument('--target_dir', type=str,
                        default='/mnt/e/DataSets/Chemistry/ForwardPrediction/DPO',
                        help='目标目录路径')
    parser.add_argument('--target_file', type=str,
                        default='qwen205_moltrans_mit_mixed_augm_dpo_lora_para1_data.json',
                        help='目标文件名')

    args = parser.parse_args()
    main()
