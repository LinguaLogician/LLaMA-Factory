# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_merge.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/20 21:33

import json
import os
from pathlib import Path

# 设置数据目录
data_dir = "/mnt/e/DataSets/Chemistry/ChemicalMechanism"
folders = ["processed", "processed1", "processed2"]

# 获取所有文件夹中的文件列表
file_lists = {}
for folder in folders:
    folder_path = os.path.join(data_dir, folder)
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        file_lists[folder] = files
    else:
        print(f"警告: 文件夹 {folder_path} 不存在")

# 找出所有唯一的文件名
all_files = set()
for files in file_lists.values():
    all_files.update(files)

# 合并相同文件名的JSON文件
for filename in all_files:
    combined_data = []

    # 从每个文件夹中读取相同文件名的文件
    for folder in folders:
        file_path = os.path.join(data_dir, folder, filename)

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_data.extend(data)
                    else:
                        print(f"警告: {file_path} 不是JSON数组，跳过")
            except Exception as e:
                print(f"错误: 读取文件 {file_path} 时出错: {e}")
        else:
            print(f"警告: 文件 {file_path} 不存在")

    # 将合并后的数据保存到data_dir目录下
    if combined_data:
        output_path = os.path.join(data_dir, filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            print(f"成功合并并保存: {output_path}")
        except Exception as e:
            print(f"错误: 保存文件 {output_path} 时出错: {e}")
    else:
        print(f"警告: 文件 {filename} 没有有效数据可合并")

print("合并完成!")
