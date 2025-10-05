# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: change_data_info.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/27 22:18
# https://chat.deepseek.com/a/chat/s/c20337f8-437c-4e47-bd7d-12f2a6e28880

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


def find_json_files(root_dir: str) -> List[Tuple[str, str]]:
    """
    查找指定目录下所有子文件夹中的.json文件（不包括根目录本身）

    Args:
        root_dir: 根目录路径

    Returns:
        List[Tuple[文件相对路径, 文件完整路径]]
    """
    json_files = []
    root_path = Path(root_dir)

    # 遍历所有子目录（不包括根目录本身）
    for file_path in root_path.rglob("*.json"):
        # 确保文件在子目录中，不在根目录
        if file_path.parent != root_path:
            # 获取相对于根目录的相对路径
            relative_path = file_path.relative_to(root_path)
            json_files.append((str(relative_path), str(file_path)))

    return json_files


def generate_key(filename: str, is_val: bool, is_augm_x1: bool) -> str:
    """
    根据规则生成字典的key

    Args:
        filename: 文件名（不含路径和扩展名）
        is_val: 是否来自val目录
        is_augm_x1: 是否来自augm_x1文件夹

    Returns:
        生成的key
    """
    # 处理文件名，去除.json扩展名
    if filename.endswith('.json'):
        filename = filename[:-5]

    # 如果来自augm_x1，添加_x1后缀
    key1 = f"{filename}_x1" if is_augm_x1 else filename

    # 如果来自val，添加_val后缀
    key = f"{key1}_val" if is_val else key1

    return key


def update_dataset_info(data_train_dir: str, data_val_dir: str, data_info_file: str) -> Dict:
    """
    更新数据集信息文件

    Args:
        data_train_dir: 训练数据目录
        data_val_dir: 验证数据目录
        data_info_file: 数据集信息文件路径

    Returns:
        更新后的数据集信息字典
    """
    print("=" * 60)
    print("开始更新数据集信息文件")
    print(f"训练数据目录: {data_train_dir}")
    print(f"验证数据目录: {data_val_dir}")
    print(f"信息文件: {data_info_file}")
    print("=" * 60)

    # 读取现有的数据集信息文件
    if os.path.exists(data_info_file):
        with open(data_info_file, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        print(f"已加载现有信息文件，包含 {len(dataset_info)} 个条目")
    else:
        dataset_info = {}
        print("信息文件不存在，将创建新文件")

    # 查找训练和验证目录下的所有json文件
    print("\n搜索训练数据目录中的JSON文件...")
    train_files = find_json_files(data_train_dir)
    print(f"找到 {len(train_files)} 个训练JSON文件")

    print("\n搜索验证数据目录中的JSON文件...")
    val_files = find_json_files(data_val_dir)
    print(f"找到 {len(val_files)} 个验证JSON文件")

    # 处理训练文件
    print("\n处理训练文件...")
    train_added = 0
    for relative_path, full_path in tqdm(train_files, desc="训练文件"):
        # 检查是否来自augm_x1文件夹
        is_augm_x1 = "augm_x1" in relative_path

        # 从相对路径中提取文件名（不含扩展名）
        filename = Path(relative_path).stem

        # 生成key
        key = generate_key(filename, is_val=False, is_augm_x1=is_augm_x1)

        # 如果key不存在，则添加到字典中
        if key not in dataset_info:
            train_added += 1
        dataset_info[key] = {"file_name": full_path.replace('/mnt/e/', '')}
    print(f"训练文件处理完成，新增 {train_added} 个条目")

    # 处理验证文件
    print("\n处理验证文件...")
    val_added = 0
    for relative_path, full_path in tqdm(val_files, desc="验证文件"):
        # 检查是否来自augm_x1文件夹
        is_augm_x1 = "augm_x1" in relative_path

        # 从相对路径中提取文件名（不含扩展名）
        filename = Path(relative_path).stem

        # 生成key
        key = generate_key(filename, is_val=True, is_augm_x1=is_augm_x1)

        # 如果key不存在，则添加到字典中
        if key not in dataset_info:
            dataset_info[key] = {"file_name": full_path}
            val_added += 1

    print(f"验证文件处理完成，新增 {val_added} 个条目")

    # 保存更新后的信息文件
    print(f"\n保存更新后的信息文件到 {data_info_file}")
    os.makedirs(os.path.dirname(data_info_file), exist_ok=True)

    with open(data_info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("更新完成!")
    print(f"信息文件现在包含 {len(dataset_info)} 个条目")
    print(f"新增训练条目: {train_added}")
    print(f"新增验证条目: {val_added}")
    print(f"文件已保存: {data_info_file}")
    print("=" * 60)

    return dataset_info


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='更新数据集信息文件')
    parser.add_argument('--data_train_dir', type=str,
                        default='/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/train',
                        help='训练数据目录路径')
    parser.add_argument('--data_val_dir', type=str,
                        default='/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/val',
                        help='验证数据目录路径')
    parser.add_argument('--data_info_file', type=str,
                        default='data/dataset_info.json',
                        help='数据集信息文件路径')

    args = parser.parse_args()

    # 检查目录是否存在
    if not os.path.exists(args.data_train_dir):
        print(f"错误: 训练数据目录不存在: {args.data_train_dir}")
        return

    if not os.path.exists(args.data_val_dir):
        print(f"错误: 验证数据目录不存在: {args.data_val_dir}")
        return

    # 更新数据集信息
    update_dataset_info(
        data_train_dir=args.data_train_dir,
        data_val_dir=args.data_val_dir,
        data_info_file=args.data_info_file
    )


if __name__ == "__main__":
    main()
