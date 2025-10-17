# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: batch_predict_score.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/28 16:46
# https://chat.deepseek.com/a/chat/s/59ced693-1ead-42ca-aee4-0a9189614201
# https://chat.deepseek.com/a/chat/s/b3b2fe88-b95f-4540-aec8-f5666a97faa0
import json
# -*- coding: utf-8 -*-
# @filename: batch_predict_score.py
# @author: Assistant
# @contact:
# @time: 2025/9/28

import os
import sys
import subprocess
import argparse
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

import GPUtil

# 添加当前目录到Python路径，以便导入predict2和score2
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入预测和评分模块
try:
    from predict2 import main as predict_main
    from score2 import main as score_main
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保predict2.py和score2.py在当前目录下")
    sys.exit(1)

DEFAULT_SUBSET=""
GPU_MEMORY_THRESHOLD = 22000  # MB，GPU显存阈值
WAIT_FOR_GPU= False

# "RXTS_TO_MECH": [
#     "UPD.CANO.STD.RXTS->CLS",
#     "ORI.CANO.STD.RXTS->CLS",
# ],
# "RXN_TO_MECH": [
#     "UPD.CANO.STD.RXN->CLS",
# ]

TASKS_CONFIG = {
    "RXN_TO_MECH": [
        "RXN->CLS",
        "RXN->MECH",
        "RXN->CLS+MECH",

        "ORI.CANO.STD.RXN->CLS", ##
        "ORI.CANO.AM.RXN->CLS", ##
        "UPD.CANO.AM.RXN->CLS", ##
        "UPD.CANO.AM.RXN->MECH", ##
        "UPD.CANO.AM.RXN->CLS+MECH", ##

        "UPD.CANO.STD.RXN->CLS",

        "ORI.ARBI.STD.RXN->CLS", ##
        "UPD.ARBI.STD.RXN->CLS", ##
    ],

    "RXN_TO_RXN": [

        "ORI.RXN->UPD.RXN",
        "STD.RXN->AM.RXN",
        "AM.RXN->STD.RXN",

        "ARBI.RXN->CANO.RXN",
        "CANO.RXN->ARBI.RXN",

        "ORI.CANO.AM.RXN->UPD.CANO.AM.RXN", ##
        "ORI.CANO.STD.RXN->UPD.CANO.STD.RXN",  ##

        "ORI.CANO.STD.RXN->ORI.CANO.AM.RXN",  ##
        "UPD.CANO.STD.RXN->UPD.CANO.AM.RXN",  ##

        "ORI.CANO.AM.RXN->ORI.CANO.STD.RXN", ##
        "UPD.CANO.AM.RXN->UPD.CANO.STD.RXN",  ##

        "UPD.CANO.AM.RXN->ORI.CANO.AM.RXN",

        "ORI.ARBI.STD.RXN->ORI.CANO.STD.RXN", ##
        "UPD.ARBI.STD.RXN->UPD.CANO.STD.RXN",  ##

        "ORI.CANO.STD.RXN->ORI.ARBI.STD.RXN",  ##
        "UPD.CANO.STD.RXN->UPD.ARBI.STD.RXN",  ##
    ],

    "RXTS_TO_RXTS": [

        "STD.RXTS->AM.RXTS",
        "AM.RXTS->STD.RXTS",
        "ARBI.RXTS->CANO.RXTS",
        "CANO.RXTS->ARBI.RXTS",

        "ORI.CANO.STD.RXTS->UPD.CANO.STD.RXTS",
        "UPD.CANO.STD.RXTS->ORI.CANO.STD.RXTS",
        "UPD.CANO.AM.RXTS->ORI.CANO.AM.RXTS",
        "ORI.CANO.AM.RXTS->UPD.CANO.AM.RXTS",


        "UPD.CANO.STD.RXTS->UPD.CANO.AM.RXTS", ##
        "UPD.CANO.AM.RXTS->UPD.CANO.STD.RXTS", ##
        "ORI.CANO.STD.RXTS->ORI.CANO.AM.RXTS", ##
        "ORI.CANO.AM.RXTS->ORI.CANO.STD.RXTS", ##

        "ORI.CANO.STD.RXTS->ORI.ARBI.STD.RXTS", ##
        "ORI.ARBI.STD.RXTS->ORI.CANO.STD.RXTS", ##
        "UPD.CANO.STD.RXTS->UPD.ARBI.STD.RXTS", ##
        "UPD.ARBI.STD.RXTS->UPD.CANO.STD.RXTS" ##
    ],
    "RXTS_TO_MECH": [
        "UPD.CANO.AM.RXTS->MECH",
        "UPD.CANO.AM.RXTS->CLS",
        "ORI.CANO.AM.RXTS->CLS",
        "UPD.CANO.AM.RXTS->CLS+MECH",

        "UPD.CANO.STD.RXTS->CLS",
        "ORI.CANO.STD.RXTS->CLS",

    ],
    "PRDS_TO_PRDS": [

        "AM.PRDS->STD.PRDS",
        "ARBI.PRDS->CANO.PRDS",
        "CANO.PRDS->ARBI.PRDS",

        "UPD.CANO.AM.PRDS->UPD.CANO.STD.PRDS", ##
        "ORI.CANO.AM.PRDS->ORI.CANO.STD.PRDS", ## //

        "ORI.CANO.STD.PRDS->ORI.ARBI.STD.PRDS", ##
        "ORI.ARBI.STD.PRDS->ORI.CANO.STD.PRDS", ##
        "UPD.CANO.STD.PRDS->UPD.ARBI.STD.PRDS", ##
        "UPD.ARBI.STD.PRDS->UPD.CANO.STD.PRDS" ##
    ],
    "RXTS_TO_PRDS": [
        "RXTS->PRDS",
        "UPD.CANO.AM.RXTS->UPD.CANO.AM.PRDS",
        "UPD.CANO.STD.RXTS->UPD.CANO.STD.PRDS", ##
        "ORI.CANO.AM.RXTS->ORI.CANO.AM.PRDS", ## //
        "ORI.CANO.STD.RXTS->ORI.CANO.STD.PRDS", ##
    ],
    "PRDS_TO_RXTS": [
        "PRDS->RXTS",
        "UPD.CANO.STD.PRDS->UPD.CANO.STD.RXTS", ##
        "ORI.CANO.STD.PRDS->ORI.CANO.STD.RXTS", ##
    ],

    "RXTS_TO_PRDS_PLUS": [
        "UPD.CANO.AM.RXTS->MECH+UPD.CANO.AM.PRDS", ##
        "UPD.CANO.AM.RXTS->CLS+UPD.CANO.AM.PRDS", ##
        "UPD.CANO.AM.RXTS->CLS+MECH+UPD.CANO.AM.PRDS", ##
        "UPD.CANO.AM.RXTS+MECH->UPD.CANO.AM.PRDS", ##
        "UPD.CANO.AM.RXTS+CLS->UPD.CANO.AM.PRDS", ##
        "UPD.CANO.AM.RXTS+CLS->MECH+UPD.CANO.AM.PRDS", ##
        "UPD.CANO.AM.RXTS+MECH->CLS+UPD.CANO.AM.PRDS", ##
        "UPD.CANO.AM.RXTS+MECH+CLS->UPD.CANO.AM.PRDS" ##
    ],

    "STYLE_TO_STYLE": [
      "CANO->ARBI",
      "ARBI->CANO",
      "STD->AM",
      "AM->STD",
    ]
}

def wait_for_gpu_memory(threshold_mb: int = 8000):
    """等待GPU显存达到阈值"""
    while True:
        available_memory1 = min([gpu.memoryFree for gpu in GPUtil.getGPUs()])
        if available_memory1 <= threshold_mb:
            print(f"GPU memory available: {available_memory1}MB")
            time.sleep(60)
            continue
        time.sleep(60*10)
        available_memory2 = min([gpu.memoryFree for gpu in GPUtil.getGPUs()])
        if available_memory2 >= threshold_mb:
            print(f"GPU memory available: {available_memory2}MB")
            break
        else:
            print(f"Waiting for GPU memory... (available: {available_memory2}MB, required: {threshold_mb}MB)")
            time.sleep(60)

def search_group(task_id):
    group = None
    task_id = task_id.lower()
    for grp, tasks in TASKS_CONFIG.items():
        for task_tag in tasks:
            # 转换task_tag为task_id格式进行比较
            task_tag_id = task_tag.replace('->', '_TO_').replace('+', '_').replace('.', '').lower()
            if task_tag_id == task_id:
                group = grp.lower()
                break
        if group:
            break

    if not group:
        raise ValueError(f"无法找到task_id {task_id} 对应的group")
    return group

class BatchChemMechProcessor:
    """批量化学机制预测和评分处理器"""

    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.tasks = self.parse_tasks()

    def setup_logging(self):
        """设置日志"""
        log_dir = Path("./logs/chemechpred/batch")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"batch_processing_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)


    # 修改 parse_tasks 方法
    def parse_tasks(self) -> List[Tuple[str, str, str]]:
        """解析任务列表"""
        tasks = []

        # 优先从命令行参数解析任务
        if self.args.tasks:
            for task_str in self.args.tasks:
                try:
                    section, model_name, task_id = task_str.split(',')
                    tasks.append((section.strip(), model_name.strip(), task_id.strip()))
                except ValueError:
                    self.logger.error(f"任务格式错误: {task_str}，应为 'section,model_name,task_id'")

        # 从配置文件解析任务
        elif self.args.config_files:
            for config_file in self.args.config_files:
                try:
                    config_tasks = load_config_from_file(config_file)
                    tasks.extend(config_tasks)
                    self.logger.info(f"从配置文件 {config_file} 加载了 {len(config_tasks)} 个任务")
                except Exception as e:
                    self.logger.error(f"加载配置文件 {config_file} 失败: {e}")

        # 从单个配置文件解析任务
        elif self.args.config_file:
            try:
                tasks = load_config_from_file(self.args.config_file)
                self.logger.info(f"从配置文件 {self.args.config_file} 加载了 {len(tasks)} 个任务")
            except Exception as e:
                self.logger.error(f"加载配置文件 {self.args.config_file} 失败: {e}")

        # 使用默认任务列表
        else:
            tasks = predict_tasks

        if not tasks:
            self.logger.warning("没有找到任何任务配置")

        return tasks

    def run_predict(self, section: str, model_name: str, task_id: str) -> bool:
        """运行单个预测任务"""
        self.logger.info(f"开始预测任务 - section: {section}, model: {model_name}, task: {task_id}")

        try:
            # 构建预测参数
            predict_args = [
                "--task_id", task_id,
                "--section", section,
                "--model_name", model_name,
                "--group", search_group(task_id),
                "--data_base_dir", self.args.data_base_dir,
                "--subset", self.args.subset,
                "--output_base_dir", self.args.output_base_dir,
                "--model_path", self.args.model_path,
                "--finetuning_type", self.args.finetuning_type,
                "--template", self.args.template,
                "--num_beams", str(self.args.num_beams),
                "--do_sample" if self.args.do_sample else "",
                "--max_new_tokens", str(self.args.max_new_tokens),
                "--num_return_sequences", str(self.args.num_return_sequences),
                "--output_scores" if self.args.output_scores else "",
                "--return_dict_in_generate" if self.args.return_dict_in_generate else "",
                "--batch_limit", str(self.args.batch_limit),
                "--batch_token_size", str(self.args.batch_token_size),
                "--minmax_gap", str(self.args.minmax_gap),
                "--wait_for_gpu", False,
                "--gpu_threshold", str(self.args.gpu_threshold)
            ]

            # 过滤空参数
            # predict_args = [arg for arg in predict_args if arg]

            # 设置sys.argv并调用predict_main
            original_argv = sys.argv
            sys.argv = ['predict2.py'] + predict_args

            predict_main()

            # 恢复原始argv
            sys.argv = original_argv

            self.logger.info(f"预测任务完成 - section: {section}, model: {model_name}, task: {task_id}")
            return True

        except Exception as e:
            self.logger.error(f"预测任务失败 - section: {section}, model: {model_name}, task: {task_id}: {e}")
            return False

    def run_score(self, section: str, model_name: str, task_id: str) -> bool:
        """运行单个评分任务"""
        self.logger.info(f"开始评分任务 - section: {section}, model: {model_name}, task: {task_id}")

        try:
            # 构建评分参数
            score_args = [
                "--group", search_group(task_id),
                "--task_id", task_id.lower(),
                "--model_name", model_name,
                "--prediction_base_dir", self.args.output_base_dir,
                "--output_base_dir", self.args.score_base_dir,
                "--subset", self.args.subset,
                "--max_k", str(self.args.max_k)
            ]

            # 设置sys.argv并调用score_main
            original_argv = sys.argv
            sys.argv = ['score2.py'] + score_args

            score_main()

            # 恢复原始argv
            sys.argv = original_argv

            self.logger.info(f"评分任务完成 - section: {section}, model: {model_name}, task: {task_id}")
            return True

        except Exception as e:
            self.logger.error(f"评分任务失败 - section: {section}, model: {model_name}, task: {task_id}: {e}")
            return False

    def process_single_task(self, section: str, model_name: str, task_id: str) -> bool:
        """处理单个任务（预测+评分）"""
        self.logger.info(f"处理任务: section={section}, model={model_name}, task={task_id}")

        # 运行预测
        predict_success = self.run_predict(section, model_name, task_id)
        if not predict_success:
            return False

        # 等待一段时间，确保文件写入完成
        time.sleep(2)

        # 运行评分
        score_success = self.run_score(section, model_name, task_id)

        return score_success

    def process_all_tasks(self):
        """处理所有任务"""
        self.logger.info(f"开始批量处理，共 {len(self.tasks)} 个任务")

        success_count = 0
        failed_tasks = []

        for i, (section, model_name, task_id) in enumerate(self.tasks, 1):
            self.logger.info(f"处理任务 {i}/{len(self.tasks)}")

            success = self.process_single_task(section, model_name, task_id)

            if success:
                success_count += 1
                self.logger.info(f"任务 {i} 处理成功")
            else:
                failed_tasks.append((section, model_name, task_id))
                self.logger.error(f"任务 {i} 处理失败")

            # 任务间延迟，避免资源冲突
            if i < len(self.tasks):
                self.logger.info("等待10秒后处理下一个任务...")
                time.sleep(10)

        # 输出总结
        self.logger.info("=" * 50)
        self.logger.info("批量处理完成")
        self.logger.info(f"成功: {success_count}/{len(self.tasks)}")

        if failed_tasks:
            self.logger.info("失败的任务:")
            for task in failed_tasks:
                self.logger.info(f"  - section: {task[0]}, model: {task[1]}, task: {task[2]}")
        else:
            self.logger.info("所有任务都成功完成！")


def load_config_from_file(config_path: str) -> List[Tuple[str, str, str]]:
    """从文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    result = []
    task_type = config.get("task_type", )

    if task_type == "multi_task":
        for model_data in config.get("tasks", []):
            section = model_data["section"]
            model = model_data["model"]
            tasks = model_data["tasks"]
            for task in tasks:
                result.append((section, model, task))
    else:
        for section_data in config.get("tasks", []):
            section = section_data.get("section", "")
            models = section_data.get("models", [])
            for model_name in models:
                if "_x" in model_name:
                    task_id = model_name.split("_x")[0]
                else:
                    task_id = model_name
                result.append((section, model_name, task_id))

    return result

def main():
    parser = argparse.ArgumentParser(description="批量化学机制预测和评分脚本")

    # 任务参数
    parser.add_argument("--tasks", nargs='+', type=str,
                        help="任务列表，格式: 'section,model_name,task_id'，例如: 'prds_to_prds,model1,task1'")
    parser.add_argument("--config_files", nargs='+', type=str,
                        help="配置文件路径列表，可以传入多个JSON文件")
    parser.add_argument("--config_file", type=str,
                        help="单个配置文件路径")
    # 路径参数
    parser.add_argument("--data_base_dir", type=str,
                        default="DataSets/Chemistry/ChemicalMechanism/via_random/test/",
                        help="数据基础目录")
    parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET,
                        help="数据子集")
    parser.add_argument("--output_base_dir", type=str,
                        default="results/chemechpred/prediction",
                        help="预测结果输出基础目录")
    parser.add_argument("--score_base_dir", type=str,
                        default="results/chemechpred/scores",
                        help="评分结果输出基础目录")
    parser.add_argument("--model_path", type=str,
                        default="chemechpred/",
                        help="模型路径")

    # 预测参数
    parser.add_argument("--finetuning_type", type=str, default="full",
                        help="微调类型")
    parser.add_argument("--template", type=str, default="qwen",
                        help="模板")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="beam数量")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="是否采样")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="最大新生成token数")
    parser.add_argument("--num_return_sequences", type=int, default=5,
                        help="返回序列数量")
    parser.add_argument("--output_scores", action="store_true", default=True,
                        help="是否输出分数")
    parser.add_argument("--return_dict_in_generate", action="store_true", default=True,
                        help="是否在生成时返回字典")

    # 批处理参数
    parser.add_argument("--batch_limit", type=int, default=3,
                        help="批量大小限制")
    parser.add_argument("--batch_token_size", type=int, default=2000,
                        help="批量token大小")
    parser.add_argument("--minmax_gap", type=int, default=40,
                        help="最小最大长度差距")

    # GPU参数
    parser.add_argument("--wait_for_gpu", type=bool, default=WAIT_FOR_GPU,
                        help="是否等待GPU内存")
    parser.add_argument("--gpu_threshold", type=int, default=GPU_MEMORY_THRESHOLD,
                        help="GPU内存阈值(MB)")

    # 评分参数
    parser.add_argument("--max_k", type=int, default=5,
                        help="最大K值")

    args = parser.parse_args()

    try:
        if args.wait_for_gpu:
            wait_for_gpu_memory(args.gpu_threshold)
        processor = BatchChemMechProcessor(args)
        processor.process_all_tasks()

    except Exception as e:
        logging.error(f"批量处理失败: {str(e)}")
        raise


if __name__ == "__main__":

    # DEFAULT_SUBSET="_random313"
    DEFAULT_SUBSET=""
    GPU_MEMORY_THRESHOLD = 22000  # MB，GPU显存阈值
    WAIT_FOR_GPU= True
    # predict_tasks_file="application/eval/_config/single_task/tasks_v3.json"
    # predict_tasks_file="application/eval/_config/single_task/tasks_v4.json"
    # predict_tasks_file="application/eval/_config/single_task/tasks_v5.json"
    # predict_tasks_file="application/eval/_config/multi_task/enhc_rxts_to_prds_v1_1.json"
    # predict_tasks_file="application/chemech/eval/_config/multi_task/vaguely_defined_v1.json"
    # predict_tasks_file="application/chemech/eval/_config/multi_task/enhc_rxts_to_prds_v4_1_and_v5_1.json"
    # predict_tasks_file="application/chemech/eval/_config/single_task/rxts_to_mech_v1.json"
    # predict_tasks_file="application/chemech/eval/_config/single_task/rxts_to_mech_v1.json"
    # predict_tasks_file="application/chemech/eval/_config/single_task/updcanostdrxts_to_updcanoamrxts.json"
    predict_tasks_file="application/chemech/eval/_config/multi_task/enhc_prds_to_rxts_v1_1.json"
    predict_tasks = load_config_from_file(predict_tasks_file)
    main()
