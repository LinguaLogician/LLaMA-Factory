# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: change_file_name.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/24 22:35
# https://chat.deepseek.com/a/chat/s/75fe5b30-2152-448d-aef0-9b202793782e
# !/usr/bin/env python3
"""
文件名批量替换工具
将指定目录中所有文件夹和文件名中的'rtxs'替换为'rxts'
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import sys


def rename_files_and_folders(data_dir, old_str='rtxs', new_str='rxts'):
    """
    递归遍历目录，将文件夹和文件名中的指定字符串进行替换

    Args:
        data_dir (str): 要处理的根目录路径
        old_str (str): 要替换的旧字符串
        new_str (str): 替换后的新字符串
    """
    # 转换为Path对象
    data_path = Path(data_dir)

    # 检查目录是否存在
    if not data_path.exists():
        print(f"错误: 目录 '{data_dir}' 不存在")
        return False

    if not data_path.is_dir():
        print(f"错误: '{data_dir}' 不是一个目录")
        return False

    print(f"开始处理目录: {data_dir}")
    print(f"替换规则: '{old_str}' -> '{new_str}'")
    print("-" * 50)

    # 首先收集所有需要重命名的路径（先处理深层路径，避免父目录重命名影响子路径访问）
    all_paths = []

    # 使用os.walk收集所有文件和文件夹路径
    for root, dirs, files in os.walk(data_dir, topdown=False):  # topdown=False确保先处理子目录
        # 添加文件夹路径
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            all_paths.append(dir_path)

        # 添加文件路径
        for file_name in files:
            file_path = Path(root) / file_name
            all_paths.append(file_path)

    print(f"找到 {len(all_paths)} 个待检查的路径")

    # 过滤出需要重命名的路径
    paths_to_rename = []
    for path in all_paths:
        if old_str in path.name:
            paths_to_rename.append(path)

    print(f"需要重命名的路径数量: {len(paths_to_rename)}")

    if len(paths_to_rename) == 0:
        print("没有找到需要重命名的文件或文件夹")
        return True

    # 显示进度条进行重命名
    success_count = 0
    error_count = 0

    with tqdm(total=len(paths_to_rename), desc="重命名进度", unit="item") as pbar:
        for old_path in paths_to_rename:
            try:
                # 生成新文件名
                new_name = old_path.name.replace(old_str, new_str)
                new_path = old_path.parent / new_name

                # 如果新路径已存在，跳过
                if new_path.exists():
                    print(f"警告: 目标路径已存在，跳过: {new_path}")
                    pbar.update(1)
                    continue

                # 执行重命名
                old_path.rename(new_path)

                # 更新成功计数
                success_count += 1
                pbar.set_postfix({'当前处理': old_path.name[:20] + '...' if len(old_path.name) > 20 else old_path.name})
                pbar.update(1)

            except Exception as e:
                error_count += 1
                print(f"错误: 重命名 {old_path} 时发生异常: {e}")
                pbar.update(1)

    # 输出结果统计
    print("-" * 50)
    print("重命名操作完成!")
    print(f"成功重命名: {success_count} 个")
    print(f"失败重命名: {error_count} 个")

    return error_count == 0


def main():
    """主函数，处理命令行参数并执行重命名操作"""
    parser = argparse.ArgumentParser(description='批量重命名文件和文件夹中的字符串')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/e/Development/LLMSpace/LLaMA-Factory/examples/train_full/chemechpred',
                        help='要处理的根目录路径')
    parser.add_argument('--old_str', type=str, default='rtxs',
                        help='要替换的旧字符串')
    parser.add_argument('--new_str', type=str, default='rxts',
                        help='替换后的新字符串')

    args = parser.parse_args()

    # 执行重命名操作
    success = rename_files_and_folders(
        data_dir=args.data_dir,
        old_str=args.old_str,
        new_str=args.new_str
    )

    # 根据操作结果返回适当的退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
