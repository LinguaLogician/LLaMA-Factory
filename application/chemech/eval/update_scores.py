# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: update_scores.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/8 19:08
# https://chat.deepseek.com/a/chat/s/a6dd022e-8bec-4793-8c01-1777d5b30392
# -*- coding: utf-8 -*-
import argparse
import json
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_prediction_files(prediction_base_dir, groups=None, task_ids=None, model_names=None):
    """
    查找符合条件的预测文件

    Args:
        prediction_base_dir: 预测结果基础目录
        groups: 指定的group列表，为空表示不限
        task_ids: 指定的task_id列表，为空表示不限
        model_names: 指定的model_name列表，为空表示不限

    Returns:
        list: 符合条件的文件路径列表，每个元素为(group, task_id, model_name, file_path)的元组
    """
    base_dir = Path(prediction_base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"预测基础目录不存在: {prediction_base_dir}")

    # 查找所有group目录（排除以"_random"开头的）
    group_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_random"):
            group_dirs.append(item)

    # 如果指定了groups，则进行过滤
    if groups:
        group_dirs = [d for d in group_dirs if d.name in groups]

    logger.info(f"找到 {len(group_dirs)} 个group目录: {[d.name for d in group_dirs]}")

    # 查找所有符合条件的文件
    prediction_files = []

    for group_dir in group_dirs:
        group_name = group_dir.name

        # 查找task_id目录
        task_dirs = [d for d in group_dir.iterdir() if d.is_dir()]

        # 如果指定了task_ids，则进行过滤
        if task_ids:
            task_dirs = [d for d in task_dirs if d.name in task_ids]

        for task_dir in task_dirs:
            task_id = task_dir.name

            # 查找model文件（排除以_para01.json结尾的）
            model_files = [f for f in task_dir.glob("*.json") if not f.name.endswith("_para01.json")]

            # 如果指定了model_names，则进行过滤
            if model_names:
                model_files = [f for f in model_files if f.stem in model_names]

            for model_file in model_files:
                model_name = model_file.stem
                prediction_files.append((group_name, task_id, model_name, model_file))

    return prediction_files


def run_score2_for_file(group, task_id, model_name, prediction_base_dir, output_base_dir, subset, max_k=5):
    """
    为单个文件运行score2评分计算

    Args:
        group: group名称
        task_id: task_id名称
        model_name: model_name名称
        prediction_base_dir: 预测结果基础目录
        output_base_dir: 输出结果基础目录
        subset: 子集名称
        max_k: 最大K值
    """
    try:
        # 构建命令行参数
        cmd = [
            sys.executable, "application/chemech/eval/score2.py",
            "--group", group,
            "--task_id", task_id,
            "--model_name", model_name,
            "--prediction_base_dir", prediction_base_dir,
            "--output_base_dir", output_base_dir,
            "--subset", subset,
            "--max_k", str(max_k)
        ]

        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.returncode == 0:
            logger.info(f"成功处理: {group}/{task_id}/{model_name}")
            return True
        else:
            logger.error(f"处理失败: {group}/{task_id}/{model_name}, 错误: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"子进程错误: {group}/{task_id}/{model_name}, 错误: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"未知错误: {group}/{task_id}/{model_name}, 错误: {str(e)}")
        return False


def batch_score_calculation(prediction_base_dir, output_base_dir, subset,
                            groups=None, task_ids=None, model_names=None,
                            max_k=5):
    """
    批量执行评分计算

    Args:
        prediction_base_dir: 预测结果基础目录
        output_base_dir: 输出结果基础目录
        subset: 子集名称
        groups: 指定的group列表
        task_ids: 指定的task_id列表
        model_names: 指定的model_name列表
        max_k: 最大K值

    Returns:
        dict: 处理结果统计
    """
    # 查找所有符合条件的文件
    logger.info("开始查找符合条件的预测文件...")
    prediction_files = find_prediction_files(prediction_base_dir, groups, task_ids, model_names)

    if not prediction_files:
        logger.warning("未找到任何符合条件的预测文件")
        return {"total": 0, "success": 0, "failed": 0}

    logger.info(f"找到 {len(prediction_files)} 个符合条件的文件")

    # 显示找到的文件信息
    for group, task_id, model_name, file_path in prediction_files:
        logger.info(f"  - {group}/{task_id}/{model_name}")

    # 批量处理
    success_count = 0
    failed_count = 0

    logger.info("开始批量评分计算...")
    with tqdm(total=len(prediction_files), desc="处理进度") as pbar:
        for group, task_id, model_name, file_path in prediction_files:
            success = run_score2_for_file(
                group=group,
                task_id=task_id,
                model_name=model_name,
                prediction_base_dir=prediction_base_dir,
                output_base_dir=output_base_dir,
                subset=subset,
                max_k=max_k
            )

            if success:
                success_count += 1
            else:
                failed_count += 1

            pbar.update(1)
            pbar.set_postfix({"成功": success_count, "失败": failed_count})

    # 输出统计结果
    logger.info("=" * 50)
    logger.info("批量处理完成!")
    logger.info(f"总计: {len(prediction_files)}")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {failed_count}")
    logger.info("=" * 50)

    return {
        "total": len(prediction_files),
        "success": success_count,
        "failed": failed_count
    }


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="批量化学机制预测评分计算脚本")

    # 路径参数
    parser.add_argument("--prediction_base_dir", type=str,
                        default="results/chemechpred/prediction",
                        help="预测结果基础目录")
    parser.add_argument("--output_base_dir", type=str,
                        default="results/chemechpred/scores",
                        help="评分结果输出基础目录")
    parser.add_argument("--subset", type=str, default="",
                        help="子集名称")

    # 过滤参数
    parser.add_argument("--groups", type=str, nargs="*", default=None,
                        help="指定要处理的group列表，多个用空格分隔")
    parser.add_argument("--task_ids", type=str, nargs="*", default=None,
                        help="指定要处理的task_id列表，多个用空格分隔")
    parser.add_argument("--model_names", type=str, nargs="*", default=None,
                        help="指定要处理的model_name列表，多个用空格分隔")

    # 计算参数
    parser.add_argument("--max_k", type=int, default=5,
                        help="计算的最大K值")

    args = parser.parse_args()

    # 执行批量处理
    batch_score_calculation(
        prediction_base_dir=args.prediction_base_dir,
        output_base_dir=args.output_base_dir,
        subset=args.subset,
        groups=args.groups,
        task_ids=args.task_ids,
        model_names=args.model_names,
        max_k=args.max_k
    )


if __name__ == "__main__":
    # 默认示例调用（可以通过修改这些参数来测试）

    # 示例1: 处理所有文件
    # batch_score_calculation(
    #     prediction_base_dir="results/chemechpred/prediction",
    #     output_base_dir="results/chemechpred/scores",
    #     subset="_random313",
    #     max_k=5
    # )

    # 示例2: 只处理特定的group
    # batch_score_calculation(
    #     prediction_base_dir="results/chemechpred/prediction",
    #     output_base_dir="results/chemechpred/scores",
    #     subset="_random313",
    #     groups=["prds_to_prds", "rcts_to_prds"],
    #     max_k=5
    # )

    # 示例3: 只处理特定的task_id和model_name
    # batch_score_calculation(
    #     prediction_base_dir="results/chemechpred/prediction",
    #     output_base_dir="results/chemechpred/scores",
    #     subset="_random313",
    #     task_ids=["updcanoamprds_to_updcanostdprds"],
    #     model_names=["model1", "model2"],
    #     max_k=5
    # )

    # 通过命令行参数调用
    main()