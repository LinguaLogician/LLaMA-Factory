# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_merge.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/18 22:03

import json
import os
from typing import List


def merge_json_files(data_dir: str, files_to_merge: List[str], output_filename: str):
    """
    合并指定目录下的多个JSON文件

    Args:
        data_dir: 数据目录路径
        files_to_merge: 要合并的文件名列表
        output_filename: 输出文件名
    """
    merged_data = []

    for filename in files_to_merge:
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"警告: 文件 {filepath} 不存在，跳过")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                merged_data.extend(data)
                print(f"已加载文件: {filename} (包含 {len(data)} 条记录)")
            else:
                print(f"警告: 文件 {filename} 不是JSON数组格式，跳过")

        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")
            continue

    # 保存合并后的文件
    output_path = os.path.join(data_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"合并完成! 共合并 {len(merged_data)} 条记录")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    # 配置参数
    data_dir = "/mnt/e/DataSets/Chemistry/RetroPrediction/Augmented/train/v3"

    # 指定要合并的文件列表（相对于data_dir的文件名）
    files_to_merge = [
        "ht3_inout_augm.json",
        "ht3_output_augm.json",
        "ht3_input_augm.json",
    ]

    # 输出文件名
    output_filename = "ht3_merged.json"

    # 执行合并
    merge_json_files(data_dir, files_to_merge, output_filename)
