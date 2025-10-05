# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: task_conversion.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/20 17:06
# https://chat.deepseek.com/a/chat/s/2ce5b822-3771-42bd-9409-5776101700fc

import os
import json
import argparse
import re

from tqdm import tqdm
from typing import List, Dict, Any


class ChemicalMechanismDataProcessor:
    def __init__(self, base_dir: str = "/mnt/e/DataSets/Chemistry/ChemicalMechanism",
                 version: str = "via_random", data_file: str = "mech_USPTO_ext.json"):
        self.base_dir = base_dir
        self.version = version
        self.data_file = data_file

        # 任务定义：输入部分 -> 输出部分
        self.tasks = {
            "UPDCANOAMRXTS_TO_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS"],
                "output": ["UPDCANOAMPRDS"]
            },
            "UPDCANOAMRXTS_TO_MECH_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS"],
                "output": ["MECH", "UPDCANOAMPRDS"]
            },
            "UPDCANOAMRXTS_TO_CLS_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS"],
                "output": ["CLS", "UPDCANOAMPRDS"]
            },
            "UPDCANOAMRXTS_TO_CLS_MECH_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS"],
                "output": ["CLS", "MECH", "UPDCANOAMPRDS"]
            },
            "UPDCANOAMRXTS_MECH_TO_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS", "MECH"],
                "output": ["UPDCANOAMPRDS"]
            },
            "UPDCANOAMRXTS_CLS_TO_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS", "CLS"],
                "output": ["UPDCANOAMPRDS"]
            },
            "UPDCANOAMRXTS_MECH_CLS_TO_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS", "MECH", "CLS"],
                "output": ["UPDCANOAMPRDS"]
            },
            "UPDCANOAMRXTS_CLS_TO_MECH_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS", "CLS"],
                "output": ["MECH", "UPDCANOAMPRDS"]
            },
            "UPDCANOAMRXTS_MECH_TO_CLS_UPDCANOAMPRDS": {
                "input": ["UPDCANOAMRXTS", "MECH"],
                "output": ["CLS", "UPDCANOAMPRDS"]
            }
        }

        # 字段映射
        self.field_mapping = {
            "UPDCANOAMRXTS": "amrxts_cano_in_upd",
            "UPDCANOAMPRDS": "amprds_cano_in_upd",
            "CLS": "mechanistic_class",
            "MECH": "mechanistic_label"
        }

        # 标识符显示格式映射
        self.display_mapping = {
            "UPDCANOAMRXTS": "UPD.CANO.AM.RXTS",
            "UPDCANOAMPRDS": "UPD.CANO.AM.PRDS",
            "CLS": "CLS",
            "MECH": "MECH"
        }

    def format_task_id(self, task_id: str) -> str:
        """格式化任务ID为显示格式"""
        # 分割任务ID为输入和输出部分
        if "_TO_" in task_id:
            input_part, output_part = task_id.split("_TO_")
        else:
            raise ValueError("无效的任务ID")

        input_fields = input_part.split("_")
        input_display = []
        for field in input_fields:
            if field in self.display_mapping:
                input_display.append(self.display_mapping[field])

        output_fields = output_part.split("_")
        output_display = []
        for field in output_fields:
            if field in self.display_mapping:
                output_display.append(self.display_mapping[field])

        formatted = "+".join(input_display) + "->" + "+".join(output_display)
        return formatted

    def get_field_value(self, data: Dict, field_key: str) -> str:
        """根据字段键获取数据值"""
        source_field = self.field_mapping.get(field_key)
        if not source_field:
            raise ValueError(f"未知的字段键: {field_key}")

        value = data.get(source_field, "")
        if value is None:
            return ""
        return str(value).strip()

    def process_single_data(self, original_data: Dict, task_id: str, split: str) -> Dict[str, str]:
        """处理单条数据"""
        task_config = self.tasks[task_id]

        # 构建输入部分
        input_parts = []
        for input_key in task_config["input"]:
            display_key = self.display_mapping[input_key]
            value = self.get_field_value(original_data, input_key)
            input_parts.append(f"{display_key}:\n{value}")

        input_text = "\n".join(input_parts)

        # 构建输出部分
        output_parts = []
        for output_key in task_config["output"]:
            display_key = self.display_mapping[output_key]
            value = self.get_field_value(original_data, output_key)
            output_parts.append(f"{display_key}:\n{value}")

        output_text = "\n".join(output_parts)

        id_no = re.findall(r'\d+', original_data['id'])[-1]
        # 构建结果
        result = {
            "id": f"{task_id.lower()}_{id_no}",
            "instruction": self.format_task_id(task_id),
            "input": input_text,
            "output": output_text
        }

        return result

    def process_split(self, split: str, tasks: List[str] = None):
        """处理单个split的数据"""
        if tasks is None:
            tasks = list(self.tasks.keys())

        data_dir = os.path.join(self.base_dir, self.version, split)
        input_file = os.path.join(data_dir, self.data_file)
        output_dir = data_dir

        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 输入文件不存在: {input_file}")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"处理 {split} 数据...")

        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data_list = json.load(f)

        print(f"找到 {len(original_data_list)} 条数据")

        # 为每个任务处理数据
        for task_id in tasks:
            if task_id not in self.tasks:
                print(f"警告: 未知任务 {task_id}，跳过")
                continue

            print(f"处理任务: {task_id}")
            processed_data = []

            # 处理每条数据
            for data_item in tqdm(original_data_list, desc=f"处理 {task_id}"):
                try:
                    processed_item = self.process_single_data(data_item, task_id, split)
                    processed_data.append(processed_item)
                except Exception as e:
                    print(f"处理数据 {data_item.get('id', '未知')} 时出错: {e}")
                    continue

            # 保存处理后的数据
            output_file = os.path.join(output_dir, f"{task_id.lower()}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            print(f"已保存 {len(processed_data)} 条数据到 {output_file}")

    def process_all_splits(self, splits: List[str] = None, tasks: List[str] = None):
        """处理所有split的数据"""
        if splits is None:
            splits = ["train", "val", "test"]

        if tasks is None:
            tasks = list(self.tasks.keys())

        print(f"开始处理数据，版本: {self.version}")
        print(f"处理的分割: {splits}")
        print(f"处理的任务: {tasks}")

        for split in splits:
            self.process_split(split, tasks)

        print("所有数据处理完成！")


def main():
    parser = argparse.ArgumentParser(description="化学机理数据处理脚本")
    parser.add_argument("--base_dir", type=str, default="/mnt/e/DataSets/Chemistry/ChemicalMechanism",
                        help="基础数据目录")
    parser.add_argument("--version", type=str, default="via_random",
                        help="数据版本")
    parser.add_argument("--data_file", type=str, default="mech_USPTO_ext.json",
                        help="数据文件名")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="要处理的数据分割")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["UPDCANOAMRXTS_TO_UPDCANOAMPRDS",
                                 "UPDCANOAMRXTS_TO_MECH_UPDCANOAMPRDS",
                                 "UPDCANOAMRXTS_TO_CLS_UPDCANOAMPRDS",
                                 "UPDCANOAMRXTS_TO_CLS_MECH_UPDCANOAMPRDS",
                                 "UPDCANOAMRXTS_MECH_TO_UPDCANOAMPRDS",
                                 "UPDCANOAMRXTS_CLS_TO_UPDCANOAMPRDS",
                                 "UPDCANOAMRXTS_MECH_CLS_TO_UPDCANOAMPRDS",
                                 "UPDCANOAMRXTS_CLS_TO_MECH_UPDCANOAMPRDS",
                                 "UPDCANOAMRXTS_MECH_TO_CLS_UPDCANOAMPRDS"],
                        help="要处理的任务列表")

    args = parser.parse_args()

    # 创建处理器实例
    processor = ChemicalMechanismDataProcessor(
        base_dir=args.base_dir,
        version=args.version,
        data_file=args.data_file
    )

    # 处理数据
    processor.process_all_splits(splits=args.splits, tasks=args.tasks)


if __name__ == "__main__":
    main()