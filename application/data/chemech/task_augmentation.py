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

from rdkit import Chem
from tqdm import tqdm
from rdkit import rdBase

# 禁用 RDKit 日志
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
import random

# ============================
# 常量配置区域
# ============================

# 基础路径配置
DEFAULT_DATA_DIR = "/mnt/e/DataSets/Chemistry/ChemicalMechanism/"
DEFAULT_VERSION = "via_random"
DEFAULT_SPLITS = ["train", "val", "test"]
# DEFAULT_SPLITS = ["val"]
DATA_FILE_NAME = "mech_USPTO_ext.json"

TASK_GROUPS0 = {
    "RXN_TO_MECH": [

        # "RXN->CLS",
        # "RXN->MECH",
        # "RXN->CLS+MECH",
        # --------------------------------------
        # "ORI.CANO.STD.RXN->CLS",
        # "ORI.CANO.AM.RXN->CLS",
        # "UPD.CANO.AM.RXN->CLS",
        # "UPD.CANO.AM.RXN->MECH",
        # "UPD.CANO.AM.RXN->CLS+MECH",
        # --------------------------------------
        "ORI.ARBI.STD.RXN->CLS",
        "UPD.ARBI.STD.RXN->CLS",
    ]
}

# 任务配置
TASK_GROUPS = {
    "RXN_TO_MECH": [

        # "RXN->CLS",
        # "RXN->MECH",
        # "RXN->CLS+MECH",
        # --------------------------------------
        "ORI.CANO.STD.RXN->CLS",
        "ORI.CANO.AM.RXN->CLS",
        "UPD.CANO.AM.RXN->CLS",
        "UPD.CANO.AM.RXN->MECH",
        "UPD.CANO.AM.RXN->CLS+MECH",
        # --------------------------------------
        "ORI.ARBI.STD.RXN->CLS",
        "UPD.ARBI.STD.RXN->CLS",
    ],

    "RXN_TO_RXN": [

        # "ORI.RXN->UPD.RXN",
        # "STD.RXN->AM.RXN",
        # "AM.RXN->STD.RXN",
        # "ARBI.RXN->CANO.RXN",
        # "CANO.RXN->ARBI.RXN",

        "ORI.CANO.AM.RXN->UPD.CANO.AM.RXN",
        "ORI.CANO.STD.RXN->UPD.CANO.STD.RXN",

        "ORI.CANO.STD.RXN->ORI.CANO.AM.RXN",
        "UPD.CANO.STD.RXN->UPD.CANO.AM.RXN",

        "ORI.CANO.AM.RXN->ORI.CANO.STD.RXN",
        "UPD.CANO.AM.RXN->UPD.CANO.STD.RXN",
    # --------------------------------------
        "ORI.ARBI.STD.RXN->ORI.CANO.STD.RXN",
        "UPD.ARBI.STD.RXN->UPD.CANO.STD.RXN",

        "ORI.CANO.STD.RXN->ORI.ARBI.STD.RXN",
        "UPD.CANO.STD.RXN->UPD.ARBI.STD.RXN",
    ],

    "RXTS_TO_RXTS": [

        # "STD.RXTS->AM.RXTS",
        # "AM.RXTS->STD.RXTS",
        # "ARBI.RXTS->CANO.RXTS",
        # "CANO.RXTS->ARBI.RXTS",

        "UPD.CANO.STD.RXTS->UPD.CANO.AM.RXTS",
        "UPD.CANO.AM.RXTS->UPD.CANO.STD.RXTS",
        "ORI.CANO.STD.RXTS->ORI.CANO.AM.RXTS",
        "ORI.CANO.AM.RXTS->ORI.CANO.STD.RXTS",
        # --------------------------------------
        "ORI.CANO.STD.RXTS->ORI.ARBI.STD.RXTS",
        "ORI.ARBI.STD.RXTS->ORI.CANO.STD.RXTS",
        "UPD.CANO.STD.RXTS->UPD.ARBI.STD.RXTS",
        "UPD.ARBI.STD.RXTS->UPD.CANO.STD.RXTS"
    ],

    "PRDS_TO_PRDS": [

        # "AM.PRDS->STD.PRDS",
        # "ARBI.PRDS->CANO.PRDS",
        # "CANO.PRDS->ARBI.PRDS",

        "UPD.CANO.AM.PRDS->UPD.CANO.STD.PRDS",
        "ORI.CANO.AM.PRDS->ORI.CANO.STD.PRDS",
        # --------------------------------------
        "ORI.CANO.STD.PRDS->ORI.ARBI.STD.PRDS",
        "ORI.ARBI.STD.PRDS->ORI.CANO.STD.PRDS",
        "UPD.CANO.STD.PRDS->UPD.ARBI.STD.PRDS",
        "UPD.ARBI.STD.PRDS->UPD.CANO.STD.PRDS"
    ],
    "RXTS_TO_PRDS": [
        # "RXTS->PRDS",
        "UPD.CANO.AM.RXTS->UPD.CANO.AM.PRDS",
        "UPD.CANO.STD.RXTS->UPD.CANO.STD.PRDS",
        "ORI.CANO.AM.RXTS->ORI.CANO.AM.PRDS",
        "ORI.CANO.STD.RXTS->ORI.CANO.STD.PRDS",
    ],
    "PRDS_TO_RXTS": [
        # "PRDS->RXTS",
        "UPD.CANO.STD.PRDS->UPD.CANO.STD.RXTS",
        "ORI.CANO.STD.PRDS->ORI.CANO.STD.RXTS",
    ],
    # "OTHERS": [
    #   "CANO->ARBI",
    #   "ARBI->CANO",
    #   "STD->AM",
    #   "AM->STD",
    # ]
}

# 字段映射配置
FIELD_MAPPINGS = {
    "ORI": {
        # "AM_RXTS": "amrxts_in_ori",
        # "AM_PRDS": "amprds_in_ori",
        # "AM_RXN": lambda data: f"{data['amrxts_in_ori']}>>{data['amprds_in_ori']}",
        "STD_RXTS": "rxts_cano_in_ori",
        "STD_PRDS": "prds_cano_in_ori",
        "STD_RXN": lambda data: f"{data['rxts_cano_in_ori']}>>{data['prds_cano_in_ori']}",
        "CANO_AM_RXTS": "amrxts_cano_in_ori",
        "CANO_AM_PRDS": "amprds_cano_in_ori",
        "CANO_AM_RXN": lambda data: f"{data['amrxts_cano_in_ori']}>>{data['amprds_cano_in_ori']}",
        # "ARBI_AM_RXTS": "amrxts_cano_in_ori",  # 需要增强
        # "ARBI_AM_PRDS": "amprds_cano_in_ori",  # 需要增强
        # "ARBI_AM_RXN": lambda data: f"{data['amrxts_cano_in_ori']}>>{data['amprds_cano_in_ori']}",  # 需要增强
        "ARBI_STD_RXTS": "rxts_cano_in_ori",  # 需要增强
        "ARBI_STD_PRDS": "prds_cano_in_ori",  # 需要增强
        "ARBI_STD_RXN": lambda data: f"{data['rxts_cano_in_ori']}>>{data['prds_cano_in_ori']}"  # 需要增强
    },
    "UPD": {
        # "AM_RXTS": "amrxts_in_upd",
        # "AM_PRDS": "amprds_in_upd",
        # "AM_RXN": lambda data: f"{data['amrxts_in_upd']}>>{data['amprds_in_upd']}",
        "STD_RXTS": "rxts_cano_in_upd",
        "STD_PRDS": "prds_cano_in_upd",
        "STD_RXN": lambda data: f"{data['rxts_cano_in_upd']}>>{data['prds_cano_in_upd']}",
        "CANO_AM_RXTS": "amrxts_cano_in_upd",
        "CANO_AM_PRDS": "amprds_cano_in_upd",
        "CANO_AM_RXN": lambda data: f"{data['amrxts_cano_in_upd']}>>{data['amprds_cano_in_upd']}",
        # "ARBI_AM_RXTS": "amrxts_cano_in_upd",  # 需要增强
        # "ARBI_AM_PRDS": "amprds_cano_in_upd",  # 需要增强
        # "ARBI_AM_RXN": lambda data: f"{data['amrxts_cano_in_upd']}>>{data['amprds_cano_in_upd']}",  # 需要增强
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

    def _get_field_key(self, data_type: str, field_type: str) -> tuple[str, str]:
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

    def _needs_augmentation(self, input_type: str, output_type: str) -> (bool, bool):
        """判断任务是否需要数据增强"""
        ignores = ["CLS", "MECH", "CLS+MECH"]
        input_has_arbi = ("ARBI" in input_type or
                          (input_type not in  ignores
                           and "AM" not in input_type
                           and "CANO" not in input_type))
        output_has_arbi = ("ARBI" in output_type or
                           (output_type not in ignores
                            and "CANO" not in output_type
                            and "AM" not in output_type))

        if self.to_be_augmented:
            return (input_has_arbi or output_has_arbi), (input_has_arbi or output_has_arbi)
        else:
            return False, input_has_arbi or output_has_arbi

    def _augment_smiles(self, smiles: str, num_variants=1) -> List[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles] * num_variants
            results = []
            for _ in range(num_variants):
                random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                results.append(random_smiles)
            return results
        except:
            return [smiles] * num_variants

    def _augment_data(self, in_out_data: str, data_type: str, multiplier=1) -> List[str]:
        if "ARBI" not in data_type:
            return [in_out_data] * multiplier
        if "RXN" in data_type:
            rxts_and_prds = in_out_data.split(">>",1)
            augmented_reactants = self._augment_smiles(rxts_and_prds[0], multiplier)
            augmented_products = self._augment_smiles(rxts_and_prds[1], multiplier)
            return [f"{rxt}>>{prd}" for rxt, prd in zip(augmented_reactants, augmented_products)]

        elif "RXTS" in data_type or "PRDS" in data_type:
            return self._augment_smiles(in_out_data, multiplier)
        else:
            raise ValueError(f"无效的数据类型: {data_type}")

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
        needs_aug, has_arbi = self._needs_augmentation(input_type, output_type)

        multiplier = self.multiple if needs_aug else 0

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

        if 'ARBI.AM' in task_tag:
            raise NotImplementedError("ARBI.AM 数据增强处理未实现")

        if needs_aug:
            augmented_inputs = self._augment_data(input_data, input_type, multiplier)
            augmented_outputs = self._augment_data(output_data, output_type, multiplier)
        else:
            augmented_inputs = [input_data]
            augmented_outputs = [output_data]

        # 生成结果
        for i, (inp, out) in enumerate(zip(augmented_inputs, augmented_outputs)):
            input_str, output_str = self._format_input_output(inp, out, input_type, output_type)

            result = {
                "id": f"{task_id.lower()}_{i if needs_aug else 0}_{data['id'].split('_')[-1]}",
                "instruction": self._format_instruction(input_type, output_type),
                "input": input_str,
                "output": output_str
            }
            results.append(result)

        return results

    def process_split(self, split: str) -> Dict[str, Dict[str, List[Dict]]]:
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
            grouped_results = {}
            for task_tag in task_tags:
                task_id = self._get_task_id(task_tag)

                # 检查任务是否需要处理
                input_type, output_type = self._parse_task_tag(task_tag)
                needs_aug, has_arbi = self._needs_augmentation(input_type, output_type)

                if (self.to_be_augmented and not has_arbi) or (not self.to_be_augmented and has_arbi):
                    continue

                print(f"  处理任务: {task_tag}")
                task_data = []

                input_source, input_field_key = self._get_field_key(input_type, "input")
                for data_point in tqdm(raw_data, desc=f"  {task_id}"):
                    results = self.process_single_data_point(data_point, task_tag)
                    task_data.extend(results)

                if task_data:
                    grouped_results[task_id] = task_data
                    print(f"    生成 {len(task_data)} 条数据")
            task_results[group_name] = grouped_results
        return task_results

    def save_results(self, split: str, task_results: Dict[str, Dict[str, List[Dict]]]):
        """保存处理结果"""
        output_base_dir = self.data_dir / self.version / split

        for group_name, task_data in task_results.items():
            group_dir = output_base_dir / group_name.lower()
            group_dir.mkdir(parents=True, exist_ok=True)

            for task_id, data in task_data.items():
                # 确定输出目录
                if any(self._needs_augmentation(*self._parse_task_tag(tag))[0]
                       for group in TASK_GROUPS.values() for tag in group
                       if self._get_task_id(tag) == task_id):
                    output_dir = group_dir / f"augm_x{self.multiple}"
                else:
                    output_dir = group_dir

                output_dir.mkdir(parents=True, exist_ok=True)

                # 保存数据
                output_file = output_dir / f"{task_id.lower()}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"保存 {len(data)} 条数据到: {output_file}")

    def update_summary(self, split: str, task_results: Dict[str, Dict[str, List[Dict]]]):
        """更新摘要信息"""
        if split not in self.summary:
            self.summary[split] = {}

        for group_tag, tasks in task_results.items():
            self.summary[split][group_tag] = len(tasks)

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
    parser.add_argument("--to_be_augmented", default=True,
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