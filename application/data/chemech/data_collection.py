# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: data_collection.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/29 1:14
# https://chat.deepseek.com/a/chat/s/c81192a9-d189-464c-a513-bff816c962dc

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import datetime

# 常量定义
TASKS = {
    "RXN_TO_MECH": [
        "ORI.CANO.STD.RXN->CLS",
        "ORI.CANO.AM.RXN->CLS",
        "UPD.CANO.AM.RXN->CLS",
        "UPD.CANO.AM.RXN->MECH",
        "UPD.CANO.AM.RXN->CLS+MECH",
        "ORI.ARBI.STD.RXN->CLS",
        "UPD.ARBI.STD.RXN->CLS",
    ],
    "RXN_TO_RXN": [
        "ORI.CANO.AM.RXN->UPD.CANO.AM.RXN",
        "ORI.CANO.STD.RXN->UPD.CANO.STD.RXN",

        "UPD.CANO.AM.RXN->ORI.CANO.AM.RXN",
        "UPD.CANO.STD.RXN->ORI.CANO.STD.RXN",

        "ORI.CANO.STD.RXN->ORI.CANO.AM.RXN",
        "UPD.CANO.STD.RXN->UPD.CANO.AM.RXN",
        "ORI.CANO.AM.RXN->ORI.CANO.STD.RXN",
        "UPD.CANO.AM.RXN->UPD.CANO.STD.RXN",
        "ORI.ARBI.STD.RXN->ORI.CANO.STD.RXN",
        "UPD.ARBI.STD.RXN->UPD.CANO.STD.RXN",
        "ORI.CANO.STD.RXN->ORI.ARBI.STD.RXN",
        "UPD.CANO.STD.RXN->UPD.ARBI.STD.RXN",
    ],
    "RXTS_TO_RXTS": [
        "UPD.CANO.STD.RXTS->UPD.CANO.AM.RXTS",
        "UPD.CANO.AM.RXTS->UPD.CANO.STD.RXTS",
        "ORI.CANO.STD.RXTS->ORI.CANO.AM.RXTS",
        "ORI.CANO.AM.RXTS->ORI.CANO.STD.RXTS",
        "ORI.CANO.STD.RXTS->ORI.ARBI.STD.RXTS",
        "ORI.ARBI.STD.RXTS->ORI.CANO.STD.RXTS",
        "UPD.CANO.STD.RXTS->UPD.ARBI.STD.RXTS",
        "UPD.ARBI.STD.RXTS->UPD.CANO.STD.RXTS"
    ],
    "PRDS_TO_PRDS": [
        "UPD.CANO.AM.PRDS->UPD.CANO.STD.PRDS",
        "ORI.CANO.AM.PRDS->ORI.CANO.STD.PRDS",
        "ORI.CANO.STD.PRDS->ORI.ARBI.STD.PRDS",
        "ORI.ARBI.STD.PRDS->ORI.CANO.STD.PRDS",
        "UPD.CANO.STD.PRDS->UPD.ARBI.STD.PRDS",
        "UPD.ARBI.STD.PRDS->UPD.CANO.STD.PRDS"
    ],
    "RXTS_TO_PRDS": [
        "UPD.CANO.AM.RXTS->UPD.CANO.AM.PRDS",
        "UPD.CANO.STD.RXTS->UPD.CANO.STD.PRDS",
        "ORI.CANO.AM.RXTS->ORI.CANO.AM.PRDS",
        "ORI.CANO.STD.RXTS->ORI.CANO.STD.PRDS",
    ],
    "RXTS_TO_PRDS_PLUS": [
        "UPD.CANO.AM.RXTS+MECH+CLS->UPD.CANO.AM.PRDS"
    ],
    "PRDS_TO_RXTS": [
        "UPD.CANO.STD.PRDS->UPD.CANO.STD.RXTS",
        "ORI.CANO.STD.PRDS->ORI.CANO.STD.RXTS",
    ]
}


class DatasetProcessor:
    def __init__(self, base_data_dir: str, tasks: Dict[str, List[str]]):
        self.base_data_dir = base_data_dir
        self.tasks = tasks
        self.task_id_to_group = self._build_task_id_to_group_mapping()

    def _build_task_id_to_group_mapping(self) -> Dict[str, str]:
        """构建task_id到group的映射"""
        mapping = {}
        for group, task_list in self.tasks.items():
            for task_tag in task_list:
                task_id = self._task_tag_to_task_id(task_tag)
                mapping[task_id.lower()] = group
        return mapping

    def _task_tag_to_task_id(self, task_tag: str) -> str:
        """将task_tag转换为task_id"""
        # 替换特殊字符
        task_id = task_tag.replace('.', '').replace('->', '_TO_').replace('+', '_')
        return task_id

    def _apply_mapping(self, text: str, mapping: Dict[str, str]) -> str:
        """应用映射替换文本中的标识符"""
        for old, new in mapping.items():
            text = text.replace(old, new)
        return text

    def _process_single_data_item(self, item: Dict, task_id: str, mapping: Dict[str, str],
                                  new_group: str, new_mixed_tasks: str) -> Dict:
        """处理单个数据项"""
        # 获取原始信息
        ori_tag = item.get("instruction", "")

        # 应用映射到instruction
        new_instruction = self._apply_mapping(ori_tag, mapping)

        # 应用映射到input和output
        new_input = self._apply_mapping(item.get("input", ""), mapping)
        new_output = self._apply_mapping(item.get("output", ""), mapping)

        # 构建新的id
        ori_id_parts = item["id"].split('_')
        ori_id_num = ori_id_parts[-1] if ori_id_parts else "0"
        new_id = f"{new_group}_{new_mixed_tasks}_{ori_id_num}"

        return {
            "id": new_id,
            "ori_id": item["id"],
            "ori_tag": ori_tag,
            "instruction": new_instruction,
            "input": new_input,
            "output": new_output
        }

    def _load_source_data(self, task_id: str, augm: str, split: str) -> List[Dict]:
        """加载源数据文件"""
        group = self.task_id_to_group.get(task_id.lower())
        if not group:
            raise ValueError(f"无法找到task_id '{task_id}'对应的group")

        data_file_path = Path(self.base_data_dir) / split / group.lower() / augm / f"{task_id.lower()}.json"

        if not data_file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file_path}")

        with open(data_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"从 {data_file_path} 加载了 {len(data)} 条数据")
        return data

    def process_dataset(self, data_recipe: Dict, data_output: Dict, output_base_dir: str):
        """处理数据集"""
        new_group, new_mixed_tasks = data_output["output"]
        splits = data_recipe["splits"]


        # 处理每个split
        for split in splits:
            print(f"\n处理 split: {split}")
            all_processed_data = []

            # 处理每个recipe项
            for recipe_item in data_recipe["recipe"]:
                task_id, augm, ratio, mapping = recipe_item
                print(f"  处理任务: {task_id}, 增强: {augm}, 比例: {ratio}")

                try:
                    # 加载源数据
                    source_data = self._load_source_data(task_id, augm, split)

                    # 采样
                    sample_size = int(len(source_data) * ratio)
                    if sample_size > len(source_data):
                        sample_size = len(source_data)

                    sampled_data = random.sample(source_data, sample_size)

                    # 处理数据
                    for item in tqdm(sampled_data, desc=f"处理 {task_id}"):
                        processed_item = self._process_single_data_item(
                            item, task_id, mapping, new_group, new_mixed_tasks
                        )
                        all_processed_data.append(processed_item)

                except Exception as e:
                    print(f"处理任务 {task_id} 时出错: {e}")
                    raise e
            # 创建输出目录
            random.shuffle(all_processed_data)
            output_dir = Path(output_base_dir) / split / new_group
            output_dir.mkdir(parents=True, exist_ok=True)
            # 保存处理后的数据
            output_file = output_dir / f"{new_mixed_tasks}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_processed_data, f, indent=2, ensure_ascii=False)

            print(f"保存 {len(all_processed_data)} 条数据到 {output_file}")

        # 保存摘要信息
            self._save_summary(data_recipe, data_output, output_dir, new_mixed_tasks)

    def _save_summary(self, data_recipe: Dict, data_output: Dict, output_dir: Path, new_mixed_tasks: str):
        """保存摘要信息"""
        summary_file = output_dir / f"{new_mixed_tasks}_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("数据集混合方案摘要\n")
            f.write("=" * 50 + "\n\n")

            f.write("输出配置:\n")
            f.write(f"  新group: {data_output['output'][0]}\n")
            f.write(f"  新mixed_tasks: {data_output['output'][1]}\n")
            f.write(f"  splits: {', '.join(data_output['splits'])}\n\n")

            f.write("数据源配方:\n")
            for i, recipe_item in enumerate(data_recipe["recipe"]):
                task_id, augm, ratio, mapping = recipe_item
                f.write(f"  源 {i + 1}:\n")
                f.write(f"    task_id: {task_id}\n")
                f.write(f"    augm: {augm}\n")
                f.write(f"    ratio: {ratio}\n")
                f.write(f"    mapping: {mapping}\n\n")

            f.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"摘要信息保存到: {summary_file}")


def load_config_from_file(config_path: str) -> Tuple[Dict, Dict]:
    """从文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    data_recipe = config.get("data_recipe")
    data_output = config.get("data_output")

    return data_recipe, data_output


def main():
    parser = argparse.ArgumentParser(description="处理生成数据集")
    parser.add_argument("--base_data_dir", type=str,
                        default="/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/",
                        help="基础数据目录路径")
    parser.add_argument("--output_base_dir", type=str,
                        default="/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/",
                        help="输出基础目录")
    parser.add_argument("--config_file", type=str, default=DEFAULT_CONFIG_FILE,
                        help="配置文件路径，包含data_recipe和data_output")
    parser.add_argument("--data_recipe", type=str, default=None,
                        help="data_recipe的JSON字符串")
    parser.add_argument("--data_output", type=str, default=None,
                        help="data_output的JSON字符串")

    args = parser.parse_args()

    # 加载配置
    if args.config_file:
        data_recipe, data_output = load_config_from_file(args.config_file)
    elif args.data_recipe and args.data_output:
        data_recipe = json.loads(args.data_recipe)
        data_output = json.loads(args.data_output)
        print("使用默认配置")

    # print("数据配方配置:")
    # print(json.dumps(data_recipe, indent=2))
    # print("\n数据输出配置:")
    # print(json.dumps(data_output, indent=2))

    # 创建处理器并执行
    processor = DatasetProcessor(args.base_data_dir, TASKS)
    processor.process_dataset(data_recipe, data_output, args.output_base_dir)

    print("\n数据集处理完成!")


if __name__ == "__main__":
    DEFAULT_CONFIG_FILE = "application/data/_config/enhc_rxts_to_prds/enhc_rxts_to_prds_v3.json"
    main()