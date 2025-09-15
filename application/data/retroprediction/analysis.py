# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: analysis.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/15 20:00
# https://chat.deepseek.com/a/chat/s/204d9bc9-8258-4813-8752-abec91b4ac1d

import argparse
import json
import os
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Mol
from tqdm import tqdm


def is_canonical_smiles(smiles: str) -> bool:
    """
    检查SMILES字符串是否为规范形式

    Args:
        smiles: SMILES字符串

    Returns:
        bool: 是否为规范形式
    """
    try:
        # 从SMILES创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # 生成规范SMILES
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # 比较原始SMILES与规范SMILES
        return smiles == canonical_smiles
    except Exception:
        return False


def process_retrosynthesis_data(data_dir: str, data_file: str, output_dir: str, output_file: str) -> None:
    """
    处理逆合成数据，检查SMILES的规范形式

    Args:
        data_dir: 输入数据目录
        data_file: 输入数据文件名
        output_dir: 输出目录
        output_file: 输出文件名
    """
    # 构建完整文件路径
    input_path = Path(data_dir) / data_file
    output_path = Path(output_dir) / output_file

    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"错误：输入文件不存在 {input_path}")
        return

    # 创建输出目录（如果不存在）
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在读取数据文件: {input_path}")

    # 读取JSON数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"找到 {len(data)} 条记录")
    print("开始处理SMILES规范形式检查...")

    # 处理每条记录
    processed_data = []
    for item in tqdm(data, desc="处理进度"):
        # 创建副本以避免修改原始数据
        processed_item = item.copy()

        # 检查input是否为规范形式
        input_smiles = processed_item.get("input", "")
        processed_item["input_canonical"] = is_canonical_smiles(input_smiles)

        # 检查output是否为规范形式
        output_smiles = processed_item.get("output", "")
        processed_item["output_canonical"] = is_canonical_smiles(output_smiles)

        processed_data.append(processed_item)

    print("处理完成，正在保存结果...")

    # 保存处理后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {output_path}")

    # 统计信息
    input_canonical_count = sum(1 for item in processed_data if item["input_canonical"])
    output_canonical_count = sum(1 for item in processed_data if item["output_canonical"])

    print(f"\n统计信息:")
    print(f"总记录数: {len(processed_data)}")
    print(f"input为规范形式的数量: {input_canonical_count} ({input_canonical_count / len(processed_data) * 100:.2f}%)")
    print(
        f"output为规范形式的数量: {output_canonical_count} ({output_canonical_count / len(processed_data) * 100:.2f}%)")


def main():
    """主函数，处理命令行参数并执行处理"""
    parser = argparse.ArgumentParser(description="处理逆合成数据，检查SMILES规范形式")

    # 添加命令行参数，设置默认值
    parser.add_argument('--data_dir', type=str, default='/mnt/e/DataSets/Chemistry/RetroSynthesis',
                        help='输入数据目录路径')
    parser.add_argument('--data_file', type=str, default='retrosynthesis_train.json',
                        help='输入数据文件名')
    parser.add_argument('--output_dir', type=str, default='/mnt/e/DataSets/Chemistry/RetroPrediction',
                        help='输出目录路径')
    parser.add_argument('--output_file', type=str, default='retrosynthesis_train_info.json',
                        help='输出文件名')

    # 解析参数
    args = parser.parse_args()

    print("=" * 50)
    print("逆合成数据处理工具")
    print("=" * 50)
    print(f"数据目录: {args.data_dir}")
    print(f"数据文件: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出文件: {args.output_file}")
    print("=" * 50)

    # 执行处理
    process_retrosynthesis_data(
        data_dir=args.data_dir,
        data_file=args.data_file,
        output_dir=args.output_dir,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
