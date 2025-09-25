# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: task_augmentation.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/25 1:59
# https://chat.deepseek.com/a/chat/s/6f903653-c928-496d-bbe3-c99e1ed8bb87
# https://chat.deepseek.com/a/chat/s/09b4f741-058b-4cef-a4d3-d6842d155431
# https://chat.deepseek.com/a/chat/s/c3a4142d-9a0b-466e-9acb-af239ec87504

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import random

# ============================
# 常量配置区域
# ============================

# 基础路径配置
DEFAULT_DATA_DIR = "/mnt/e/DataSets/Chemistry/ChemicalMechanism/demo"
DEFAULT_VERSION = "via_random"
DEFAULT_SPLITS = ["train", "val", "test"]
DATA_FILE_NAME = "mech_USPTO_ext.json"

# 任务配置
TASK_GROUPS = {
    "RXN-to-Mech": [
        "ORI.CANO.STD.RXN->CLS",
        "ORI.CANO.AM.RXN->CLS",
        "UPD.CANO.AM.RXN->CLS",
        "UPD.CANO.AM.RXN->MECH",
        "UPD.CANO.AM.RXN->CLS+MECH"
    ],

    "RXN-to-RXN": [
        "ORI.CANO.AM.RXN->UPD.CANO.AM.RXN",
        "ORI.CANO.STD.RXN->UPD.CANO.STD.RXN",
        "ORI.CANO.STD.RXN->ORI.CANO.AM.RXN",
        "UPD.CANO.STD.RXN->UPD.CANO.AM.RXN",
        "ORI.CANO.AM.RXN->ORI.CANO.STD.RXN",
        "UPD.CANO.AM.RXN->UPD.CANO.STD.RXN",
        "ORI.AM.RXN->UPD.AM.RXN",
        "UPD.ARBI.AM.RXN->UPD.ARBI.STD.RXN",
        "ARBI.AM.RXN->ARBI.STD.RXN"
    ],

    "Rxts-to-Rxts": [
        "UPD.CANO.STD.RXTS->UPD.CANO.AM.RXTS",
        "UPD.CANO.AM.RXTS->UPD.CANO.STD.RXTS",
        "ORI.CANO.STD.RXTS->ORI.CANO.AM.RXTS",
        "ORI.CANO.AM.RXTS->ORI.CANO.STD.RXTS",
        "AM.RXTS->AM.RXTS",
        "STD.RXTS->STD.RXTS",
        "CANO.AM.RXTS->ARBI.AM.RXTS",
        "ARBI.AM.RXTS->CANO.AM.RXTS",
        "CANO.STD.RXTS->ARBI.STD.RXTS",
        "ARBI.STD.RXTS->CANO.STD.RXTS",
        "ORI.CANO.AM.RXTS->ORI.ARBI.AM.RXTS",
        "ORI.ARBI.AM.RXTS->ORI.CANO.AM.RXTS",
        "ORI.CANO.STD.RXTS->ORI.ARBI.STD.RXTS",
        "ORI.ARBI.STD.RXTS->ORI.CANO.STD.RXTS",
        "UPD.CANO.AM.RXTS->UPD.ARBI.AM.RXTS",
        "UPD.ARBI.AM.RXTS->UPD.CANO.AM.RXTS",
        "UPD.CANO.STD.RXTS->UPD.ARBI.STD.RXTS",
        "UPD.ARBI.STD.RXTS->UPD.CANO.STD.RXTS"
    ],

    "Prds-to-Prds": [
        "UPD.CANO.AM.PRDS->UPD.CANO.STD.PRDS",
        "ORI.CANO.AM.PRDS->ORI.CANO.STD.PRDS",
        "AM.PRDS->AM.PRDS",
        "STD.PRDS->STD.PRDS",
        "CANO.AM.PRDS->ARBI.AM.PRDS",
        "ARBI.AM.PRDS->CANO.AM.PRDS",
        "CANO.STD.PRDS->ARBI.STD.PRDS",
        "ARBI.STD.PRDS->CANO.STD.PRDS",
        "ORI.CANO.AM.PRDS->ORI.ARBI.AM.PRDS",
        "ORI.ARBI.AM.PRDS->ORI.CANO.AM.PRDS",
        "ORI.CANO.STD.PRDS->ORI.ARBI.STD.PRDS",
        "ORI.ARBI.STD.PRDS->ORI.CANO.STD.PRDS",
        "UPD.CANO.AM.PRDS->UPD.ARBI.AM.PRDS",
        "UPD.ARBI.AM.PRDS->UPD.CANO.AM.PRDS",
        "UPD.CANO.STD.PRDS->UPD.ARBI.STD.PRDS",
        "UPD.ARBI.STD.PRDS->UPD.CANO.STD.PRDS"
    ],

    "Rxts-to-Prds": [
        "UPD.CANO.AM.RXTS->UPD.CANO.AM.PRDS",
        "UPD.CANO.STD.RXTS->UPD.CANO.STD.PRDS",
        "ORI.CANO.AM.RXTS->ORI.CANO.AM.PRDS",
        "ORI.CANO.STD.RXTS->ORI.CANO.STD.PRDS",
        "ARBI.AM.RXTS->ARBI.AM.PRDS",
        "ARBI.STD.RXTS->ARBI.STD.PRDS"
    ],

    "Prds-to-Rxts": [
        "UPD.CANO.STD.PRDS->UPD.CANO.STD.RXTS",
        "UPD.CANO.AM.PRDS->UPD.CANO.AM.RXTS",
        "ORI.CANO.STD.PRDS->ORI.CANO.STD.RXTS",
        "ORI.CANO.AM.PRDS->ORI.CANO.AM.RXTS",
        "ORI.ARBI.STD.PRDS->ORI.ARBI.STD.RXTS",
        "UPD.ARBI.STD.PRDS->UPD.ARBI.STD.RXTS"
    ]
}

# 字段映射配置
FIELD_MAPPINGS = {
    "ORI": {
        "AM_RXTS": "amrxts_in_ori",
        "AM_PRDS": "amprds_in_ori",
        "AM_RXN": lambda data: f"{data['amrxts_in_ori']}>>{data['amprds_in_ori']}",
        "STD_RXTS": "rxts_cano_in_ori",
        "STD_PRDS": "prds_cano_in_ori",
        "STD_RXN": lambda data: f"{data['rxts_cano_in_ori']}>>{data['prds_cano_in_ori']}",
        "CANO_AM_RXTS": "amrxts_cano_in_ori",
        "CANO_AM_PRDS": "amprds_cano_in_ori",
        "CANO_AM_RXN": lambda data: f"{data['amrxts_cano_in_ori']}>>{data['amprds_cano_in_ori']}",
        "ARBI_AM_RXTS": "amrxts_cano_in_ori",  # 需要增强
        "ARBI_AM_PRDS": "amprds_cano_in_ori",  # 需要增强
        "ARBI_AM_RXN": lambda data: f"{data['amrxts_cano_in_ori']}>>{data['amprds_cano_in_ori']}",  # 需要增强
        "ARBI_STD_RXTS": "rxts_cano_in_ori",  # 需要增强
        "ARBI_STD_PRDS": "prds_cano_in_ori",  # 需要增强
        "ARBI_STD_RXN": lambda data: f"{data['rxts_cano_in_ori']}>>{data['prds_cano_in_ori']}"  # 需要增强
    },
    "UPD": {
        "AM_RXTS": "amrxts_in_upd",
        "AM_PRDS": "amprds_in_upd",
        "AM_RXN": lambda data: f"{data['amrxts_in_upd']}>>{data['amprds_in_upd']}",
        "STD_RXTS": "rxts_cano_in_upd",
        "STD_PRDS": "prds_cano_in_upd",
        "STD_RXN": lambda data: f"{data['rxts_cano_in_upd']}>>{data['prds_cano_in_upd']}",
        "CANO_AM_RXTS": "amrxts_cano_in_upd",
        "CANO_AM_PRDS": "amprds_cano_in_upd",
        "CANO_AM_RXN": lambda data: f"{data['amrxts_cano_in_upd']}>>{data['amprds_cano_in_upd']}",
        "ARBI_AM_RXTS": "amrxts_cano_in_upd",  # 需要增强
        "ARBI_AM_PRDS": "amprds_cano_in_upd",  # 需要增强
        "ARBI_AM_RXN": lambda data: f"{data['amrxts_cano_in_upd']}>>{data['amprds_cano_in_upd']}",  # 需要增强
        "ARBI_STD_RXTS": "rxts_cano_in_upd",  # 需要增强
        "ARBI_STD_PRDS": "prds_cano_in_upd",  # 需要增强
        "ARBI_STD_RXN": lambda data: f"{data['rxts_cano_in_upd']}>>{data['prds_cano_in_upd']}"  # 需要增强
    }
}

# 特殊输出字段
SPECIAL_OUTPUTS = {
    "CLS": lambda data: data.get("mechanistic_class", ""),
    "MECH": lambda data: data.get("mechanistic_label", ""),
    "CLS+MECH": lambda data: data.get("mechanistic_class", "")+"\n"+data.get("mechanistic_label", "")
}


# ============================
# 数据处理类
# ============================

class ChemicalMechanismDataProcessor:
    def __init__(self, data_dir: str, version: str, to_be_augmented: bool = False, multiple: int = 1):
        self.data_dir = Path(data_dir)
        self.version = version
        self.to_be_augmented = to_be_augmented
        self.multiple = multiple
        self.summary = {}

    def _get_task_id(self, task_tag: str) -> str:
        """将任务标签转换为任务ID"""
        return (task_tag.replace('.', '')
                .replace('->', '_TO_')
                .replace('+', '_').upper())

    def _parse_task_tag(self, task_tag: str) -> Tuple[str, str]:
        """解析任务标签为输入和输出部分"""
        if '->' not in task_tag:
            raise ValueError(f"无效的任务标签: {task_tag}")

        input_part, output_part = task_tag.split('->')
        return input_part.strip(), output_part.strip()

    def _get_field_key(self, data_type: str, field_type: str) -> str:
        """根据数据类型和字段类型获取字段键"""
        # 解析数据类型 (如 "UPD.CANO.AM.RXTS")
        parts = data_type.split('.')

        # 确定数据源 (ORI/UPD)
        source = "UPD" if "UPD" in parts else "ORI"

        # 确定字段类型组合
        field_comb = ""
        if "ARBI" in parts:
            if "AM" in parts:
                field_comb = "ARBI_AM"
            else:
                field_comb = "ARBI_STD"
        else:  # CANO or default
            if "AM" in parts:
                field_comb = "CANO_AM" if "CANO" in parts else "AM"
            else:
                field_comb = "STD"

        # 确定数据类别
        data_category = ""
        if "RXTS" in parts:
            data_category = "RXTS"
        elif "PRDS" in parts:
            data_category = "PRDS"
        elif "RXN" in parts:
            data_category = "RXN"

        field_key = f"{field_comb}_{data_category}"
        return source, field_key

    def _get_data_value(self, data: Dict, source: str, field_key: str) -> str:
        """从数据中获取指定字段的值"""
        if field_key in SPECIAL_OUTPUTS:
            return SPECIAL_OUTPUTS[field_key](data)

        field_config = FIELD_MAPPINGS[source].get(field_key)
        if field_config is None:
            raise ValueError(f"未找到字段配置: {source}.{field_key}")

        if callable(field_config):
            return field_config(data)
        else:
            return data.get(field_config, "")

    def _needs_augmentation(self, input_type: str, output_type: str) -> bool:
        """判断任务是否需要数据增强"""
        input_has_arbi = "ARBI" in input_type or ("CANO" not in input_type and "ARBI" not in input_type)
        output_has_arbi = "ARBI" in output_type or ("CANO" not in output_type and "ARBI" not in output_type)

        if self.to_be_augmented:
            return input_has_arbi or output_has_arbi
        else:
            return False

    def _augment_smiles(self, smiles: str, num_variants: int = 1) -> List[str]:
        """对SMILES进行数据增强（生成随机SMILES）"""
        # 这里应该使用化学信息学库如RDKit来生成随机SMILES
        # 由于环境限制，这里使用简单的占位实现
        # 实际应用中应该替换为真正的随机SMILES生成

        augmented = []
        for i in range(num_variants):
            # 模拟随机SMILES生成 - 实际应该使用RDKit等
            augmented.append(f"{smiles}_augmented_{i + 1}")

        return augmented

    def _format_instruction(self, input_type: str, output_type: str) -> str:
        """格式化指令"""
        return f"{input_type}->{output_type}"

    def _format_input_output(self, input_data: str, output_data: str, input_type: str, output_type: str) -> Tuple[
        str, str]:
        """格式化输入和输出"""
        input_str = f"{input_type}:\n{input_data}"

        # 处理多输出情况
        if '+' in output_type:
            output_parts = []
            output_data = output_data.split('\n')
            for i, part in enumerate(output_type.split('+')):
                part_data = output_data[i] if part in ['CLS', 'MECH'] else ""
                output_parts.append(f"{part}:\n{part_data}")
            output_str = "\n".join(output_parts)
        else:
            output_str = f"{output_type}:\n{output_data}"

        return input_str, output_str

    def process_single_data_point(self, data: Dict, task_tag: str) -> List[Dict]:
        """处理单个数据点"""
        task_id = self._get_task_id(task_tag)
        input_type, output_type = self._parse_task_tag(task_tag)

        # 检查是否需要增强
        needs_aug = self._needs_augmentation(input_type, output_type)
        multiplier = self.multiple if needs_aug else 1

        results = []

        # 获取输入数据
        input_source, input_field_key = self._get_field_key(input_type, "input")
        input_data = self._get_data_value(data, input_source, input_field_key)

        # 获取输出数据
        if output_type in ['CLS', 'MECH', 'CLS+MECH']:
            output_data = self._get_data_value(data, "ORI", output_type)
        else:
            output_source, output_field_key = self._get_field_key(output_type, "output")
            output_data = self._get_data_value(data, output_source, output_field_key)

        if not input_data or not output_data:
            return results

        # 数据增强处理
        if needs_aug:
            augmented_inputs = self._augment_smiles(input_data, multiplier)
            augmented_outputs = self._augment_smiles(output_data, multiplier)
        else:
            augmented_inputs = [input_data]
            augmented_outputs = [output_data]

        # 生成结果
        for i, (inp, out) in enumerate(zip(augmented_inputs, augmented_outputs)):
            input_str, output_str = self._format_input_output(inp, out, input_type, output_type)

            result = {
                "id": f"{task_id.lower()}_{data['id'].split('_')[-1]}_{i if needs_aug else 0}",
                "instruction": self._format_instruction(input_type, output_type),
                "input": input_str,
                "output": output_str
            }
            results.append(result)

        return results

    def process_split(self, split: str) -> Dict[str, List[Dict]]:
        """处理单个split的数据"""
        print(f"处理 {split} 数据...")

        # 构建文件路径
        data_path = self.data_dir / self.version / split / DATA_FILE_NAME
        if not data_path.exists():
            print(f"警告: 数据文件不存在: {data_path}")
            return {}

        # 读取数据
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 处理所有任务
        task_results = {}

        for group_name, task_tags in TASK_GROUPS.items():
            for task_tag in task_tags:
                task_id = self._get_task_id(task_tag)

                # 检查任务是否需要处理
                input_type, output_type = self._parse_task_tag(task_tag)
                needs_aug = self._needs_augmentation(input_type, output_type)

                if (self.to_be_augmented and not needs_aug) or (not self.to_be_augmented and needs_aug):
                    continue

                print(f"  处理任务: {task_tag}")
                task_data = []

                for data_point in tqdm(raw_data, desc=f"  {task_id}"):
                    results = self.process_single_data_point(data_point, task_tag)
                    task_data.extend(results)

                if task_data:
                    task_results[task_id] = task_data
                    print(f"    生成 {len(task_data)} 条数据")

        return task_results

    def save_results(self, split: str, task_results: Dict[str, List[Dict]]):
        """保存处理结果"""
        output_base_dir = self.data_dir / self.version / split / "tasks"

        for task_id, data in task_results.items():
            # 确定输出目录
            if any(self._needs_augmentation(*self._parse_task_tag(tag))
                   for group in TASK_GROUPS.values() for tag in group
                   if self._get_task_id(tag) == task_id):
                output_dir = output_base_dir / f"augm_x{self.multiple}"
            else:
                output_dir = output_base_dir

            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存数据
            output_file = output_dir / f"{task_id.lower()}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"保存 {len(data)} 条数据到: {output_file}")

    def update_summary(self, split: str, task_results: Dict[str, List[Dict]]):
        """更新摘要信息"""
        if split not in self.summary:
            self.summary[split] = {}

        for task_id, data in task_results.items():
            self.summary[split][task_id] = len(data)

    def save_summary(self):
        """保存摘要信息"""
        summary_dir = self.data_dir / self.version
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary_file = summary_dir / "summary.json"

        summary_info = {
            "version": self.version,
            "to_be_augmented": self.to_be_augmented,
            "multiple": self.multiple,
            "data_stats": self.summary,
            "total_tasks": sum(len(tasks) for tasks in self.summary.values())
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, indent=2, ensure_ascii=False)

        print(f"摘要信息保存到: {summary_file}")

    def run(self, splits: List[str] = None):
        """运行数据处理流程"""
        if splits is None:
            splits = DEFAULT_SPLITS

        print(f"开始处理化学机制数据 (版本: {self.version})")
        print(f"数据增强: {self.to_be_augmented}, 倍数: {self.multiple}")
        print("=" * 50)

        for split in splits:
            task_results = self.process_split(split)
            if task_results:
                self.save_results(split, task_results)
                self.update_summary(split, task_results)
            else:
                print(f"{split} 没有生成任何数据")

        self.save_summary()
        print("数据处理完成!")


# ============================
# 主函数
# ============================

def main():
    parser = argparse.ArgumentParser(description="化学机制数据处理工具")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="数据目录根路径")
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION,
                        help="数据版本")
    parser.add_argument("--splits", type=str, nargs='+', default=DEFAULT_SPLITS,
                        help="要处理的数据分割")
    parser.add_argument("--to_be_augmented", default=False,
                        help="是否进行数据增强")
    parser.add_argument("--multiple", type=int, default=1,
                        help="数据增强倍数")

    args = parser.parse_args()

    # 创建处理器并运行
    processor = ChemicalMechanismDataProcessor(
        data_dir=args.data_dir,
        version=args.version,
        to_be_augmented=args.to_be_augmented,
        multiple=args.multiple
    )

    processor.run(args.splits)


if __name__ == "__main__":
    main()