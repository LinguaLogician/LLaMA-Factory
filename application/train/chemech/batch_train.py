# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: batch_train.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/28 18:40
# https://chat.deepseek.com/a/chat/s/bda95806-82de-424f-af66-7e6cd0f899d4
import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import yaml
import GPUtil

GPU_MEMORY_THRESHOLD = 8000  # MB，GPU显存阈值

def wait_for_gpu_memory(threshold_mb: int = GPU_MEMORY_THRESHOLD):
    """等待GPU显存达到阈值"""
    while True:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No GPU found, proceeding with CPU...")
            break

        available_memory = min([gpu.memoryFree for gpu in gpus])
        if available_memory >= threshold_mb:
            print(f"GPU memory available: {available_memory}MB")
            break
        else:
            print(f"Waiting for GPU memory... (available: {available_memory}MB, required: {threshold_mb}MB)")
            time.sleep(60)


def load_base_parameters(base_para_file: str) -> Dict[str, Any]:
    """加载基础参数文件"""
    try:
        with open(base_para_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading base parameter file {base_para_file}: {e}")
        return {}


def build_command(base_params: Dict[str, Any],
                  group: str,
                  model_name: str,
                  dataset_name: str,
                  extra_params: Dict[str, Any]) -> List[str]:
    """构建训练命令"""

    # 基础参数副本，避免修改原始数据
    params = base_params.copy()

    # 覆盖特定参数
    params.update({
        'dataset': dataset_name,
        'output_dir': f"chemechpred/{group}/{model_name}",
        'eval_dataset': f"{dataset_name}_val",
        'run_name': model_name
    })

    # 覆盖额外参数
    params.update(extra_params)

    # 构建命令参数
    cmd_args = []
    for key, value in params.items():
        if value is not None:
            if isinstance(value, bool):
                # 布尔值处理
                if value:
                    cmd_args.extend([f"--{key}"])
            elif isinstance(value, list):
                # 列表值处理
                for item in value:
                    cmd_args.extend([f"--{key}", str(item)])
            else:
                cmd_args.extend([f"--{key}", str(value)])

    # 构建完整命令
    command = ["FORCE_TORCHRUN=1", "llamafactory-cli", "train"] + cmd_args

    return command


def setup_logging(group: str, model_name: str) -> Tuple[str, str]:
    """设置日志文件路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/train/{group}"
    log_file = f"{log_dir}/{model_name}_{timestamp}.log"

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    return log_file, timestamp


def run_training(command: List[str], log_file: str) -> Tuple[bool, str]:
    """运行训练命令并记录日志"""
    try:
        # 将命令列表转换为字符串
        command_str = " ".join(command)
        print(f"Executing: {command_str}")

        # 打开日志文件
        with open(log_file, 'w', encoding='utf-8') as log:
            # 写入命令信息
            log.write(f"Command: {command_str}\n")
            log.write(f"Start time: {datetime.now().isoformat()}\n")
            log.write("=" * 80 + "\n\n")

            # 执行命令
            process = subprocess.Popen(
                command_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # 实时输出到日志文件和控制台
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='')

            # 等待进程完成
            return_code = process.wait()

            # 记录结束信息
            log.write(f"\n\nEnd time: {datetime.now().isoformat()}\n")
            log.write(f"Return code: {return_code}\n")

            success = (return_code == 0)
            return success, f"Return code: {return_code}"

    except Exception as e:
        error_msg = f"Error executing command: {e}"
        print(error_msg)
        # 确保错误信息也写入日志
        try:
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\nERROR: {error_msg}\n")
        except:
            pass
        return False, error_msg


def batch_train(training_configs: List[Tuple[str, str, str, Dict[str, Any]]],
                base_para_file: str = "examples/train_full/chemechpred/base_para.yaml",
                log_all_commands: bool = True) -> List[Dict[str, Any]]:
    """
    批量训练模型

    Args:
        training_configs: 训练配置列表，每个元素为 (group, model_name, dataset_name, extra_params)
        base_para_file: 基础参数文件路径
        log_all_commands: 是否记录所有命令到总日志

    Returns:
        训练结果列表
    """

    # 加载基础参数
    base_params = load_base_parameters(base_para_file)
    if not base_params:
        print("Warning: Base parameters are empty or failed to load")

    print(f"Loaded base parameters from: {base_para_file}")
    print(f"Number of training configurations: {len(training_configs)}")

    # 创建总日志目录
    os.makedirs("logs/train", exist_ok=True)

    # 记录所有命令的总日志
    if log_all_commands:
        total_log_file = f"logs/train/batch_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(total_log_file, 'w', encoding='utf-8') as total_log:
            total_log.write(f"Batch training started at: {datetime.now().isoformat()}\n")
            total_log.write(f"Total configurations: {len(training_configs)}\n\n")

    results = []

    for idx, (group, model_name, dataset_name, extra_params) in enumerate(training_configs):
        print(f"\n{'=' * 80}")
        print(f"Starting training {idx + 1}/{len(training_configs)}")
        print(f"Group: {group}, Model: {model_name}, Dataset: {dataset_name}")
        print(f"Extra parameters: {extra_params}")

        # 设置日志
        log_file, timestamp = setup_logging(group, model_name)

        # 构建命令
        command = build_command(base_params, group, model_name, dataset_name, extra_params)

        # 记录到总日志
        if log_all_commands:
            with open(total_log_file, 'a', encoding='utf-8') as total_log:
                total_log.write(f"\nConfiguration {idx + 1}:\n")
                total_log.write(f"  Group: {group}, Model: {model_name}, Dataset: {dataset_name}\n")
                total_log.write(f"  Command: {' '.join(command)}\n")
                total_log.write(f"  Log file: {log_file}\n")
                total_log.write(f"  Start time: {datetime.now().isoformat()}\n")

        # 执行训练
        start_time = time.time()
        success, message = run_training(command, log_file)
        end_time = time.time()
        duration = end_time - start_time

        # 记录结果
        result = {
            'index': idx,
            'group': group,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'success': success,
            'message': message,
            'duration': duration,
            'log_file': log_file,
            'timestamp': timestamp,
            'command': ' '.join(command)
        }
        results.append(result)

        # 更新总日志
        if log_all_commands:
            with open(total_log_file, 'a', encoding='utf-8') as total_log:
                status = "SUCCESS" if success else "FAILED"
                total_log.write(f"  End time: {datetime.now().isoformat()}\n")
                total_log.write(f"  Duration: {duration:.2f} seconds\n")
                total_log.write(f"  Status: {status} - {message}\n")

        print(f"Training {idx + 1} completed: {'SUCCESS' if success else 'FAILED'}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Log file: {log_file}")

    return results


def print_summary(results: List[Dict[str, Any]]):
    """打印训练结果摘要"""
    print(f"\n{'=' * 80}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 80}")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Total tasks: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFAILED TASKS:")
        for result in failed:
            print(f"  Index {result['index']}: {result['group']}/{result['model_name']}")
            print(f"    Dataset: {result['dataset_name']}")
            print(f"    Error: {result['message']}")
            print(f"    Log: {result['log_file']}")

    print(f"\nDETAILED RESULTS:")
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"  [{result['index']}] {result['group']}/{result['model_name']}: {status}")
        print(f"      Dataset: {result['dataset_name']}")
        print(f"      Duration: {result['duration']:.2f}s")
        print(f"      Log: {result['log_file']}")


def main():
    """主函数 - 支持从命令行参数或默认配置读取训练配置"""

    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        try:
            # 从命令行参数读取JSON配置
            config_json = sys.argv[1]
            training_configs = json.loads(config_json)
            # 验证配置格式
            if not all(len(item) == 4 for item in training_configs):
                raise ValueError("Invalid configuration format")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing command line arguments: {e}")
            print("Using default training configurations")
            training_configs = DEFAULT_TRAINING_CONFIGS
    else:
        print("No command line arguments provided, using default training configurations")
        training_configs = DEFAULT_TRAINING_CONFIGS

    print(f"Starting batch training with {len(training_configs)} configurations")

    if wait_for_gpu:
        wait_for_gpu_memory(gpu_threshold)


    # 执行批量训练
    results = batch_train(
        training_configs=training_configs,
        base_para_file=BASE_PARA_FILE
    )

    # 打印摘要
    print_summary(results)

    # 返回退出码（如果有失败的任务）
    if any(not result['success'] for result in results):
        sys.exit(1)


if __name__ == "__main__":
    # 默认训练配置 - 可以根据需要修改
    DEFAULT_TRAINING_CONFIGS = [
        ("enhc_rxts_to_prds", "enhc_rxts_to_prds_v1_1", "enhc_rxts_to_prds_v1_1", {}),
        # ("enhc_rxts_to_prds", "enhc_rxts_to_prds_v2_1", "enhc_rxts_to_prds_v2_1", {}),
        # ("enhc_rxts_to_prds", "enhc_rxts_to_prds_v3_1", "enhc_rxts_to_prds_v3_1", {}),
    # per_device_train_batch_size: 4
    # gradient_accumulation_steps: 1
    ]
    # 基础参数文件路径
    BASE_PARA_FILE = "examples/train_full/chemechpred/base_para.yaml"
    wait_for_gpu = True
    gpu_threshold = 10000
    main()