# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: save_to_mongo.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/1 13:51
# https://chat.deepseek.com/a/chat/s/28c1672d-8ba4-452a-bdb7-876623fe203f

import os
import json
import re
from typing import Dict, Any, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from tqdm import tqdm
import argparse


def classify_task_type(task_id: str) -> List[str]:
    """根据task_id判断任务类型"""
    task_id_lower = task_id.lower()
    pred_types = []
    if 'cls' in task_id_lower:
        pred_types.append('cls')
    if 'mech' in task_id_lower:
        pred_types.append('mech')
    if any(keyword in task_id_lower for keyword in ['rxts', 'prds', 'rxn']):
        pred_types.append('mols')
    if not pred_types:
        raise ValueError(f"无法识别任务类型: {task_id}")
    return pred_types



def transform_metrics(metrics_data: Dict[str, Any]) -> Dict[str, Any]:
    """转换评测指标格式"""
    transformed = {}

    for metric_name, k_values in metrics_data.items():
        # 首字母转小写
        new_metric_name = metric_name[0].lower() + metric_name[1:]
        transformed[new_metric_name] = {}

        for k_key, k_value in k_values.items():
            # K=1 -> k_1
            new_k_key = k_key.replace('K=', 'k_')
            transformed[new_metric_name][new_k_key] = k_value

    return transformed


def process_score_file(file_path: str, group: str, task_id: str, model_name: str) -> Dict[str, Any]:
    """处理单个评分文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 按照指定顺序创建基础信息字典
    result = {
        'group': group,
        'task': task_id.lower(),
        'model': model_name,
        'mols': {},
        'cls': {},
        'mech': {},
        'matched': {},
        'resolved': {},
        'totally': {},
        'file_path': file_path,
        'count': data.get('count', 0),
        'max_k': data.get('max_k', 5)
    }

    # 处理预测项
    pred_types = classify_task_type(task_id)

    for task_type in pred_types:
        for key, value in data.items():
            if key in ['matched', 'resolved', 'totally']:
                if 'PrecTopK' in value and 'PrecKth' in value:
                    result[key] = transform_metrics(value)

            elif task_type == 'mols' and any(keyword in key for keyword in ['prds', 'rxts', 'rxn']):
                # 处理MOLS类型的预测项
                mols_metrics = {}
                required_metrics = [
                    'AccTopK', 'AccKth', 'AccTopKMols', 'AccKthMols',
                    'PrecTopKFormat', 'PrecKthFormat', 'PrecTopKManner',
                    'PrecKthManner', 'PrecTopKValid', 'PrecKthValid'
                ]

                for metric in required_metrics:
                    if metric in value:
                        mols_metrics.update(transform_metrics({metric: value[metric]}))

                if mols_metrics:
                    result['mols'] = mols_metrics

            elif task_type == 'cls' and 'cls' in key.lower():
                # 处理CLS类型的预测项
                cls_metrics = {}
                required_metrics = ['AccTopK', 'AccKth']

                for metric in required_metrics:
                    if metric in value:
                        cls_metrics.update(transform_metrics({metric: value[metric]}))

                if cls_metrics:
                    result['cls'] = cls_metrics

            elif task_type == 'mech' and 'mech' in key.lower():
                # 处理MECH类型的预测项
                mech_metrics = {}
                required_metrics = ['AccTopK', 'AccKth']

                for metric in required_metrics:
                    if metric in value:
                        mech_metrics.update(transform_metrics({metric: value[metric]}))

                if mech_metrics:
                    result['mech'] = mech_metrics

    return result


def find_all_score_files(scores_dir: str) -> List[tuple]:
    """查找所有评分文件"""
    score_files = []

    for group in os.listdir(scores_dir):
        group_path = os.path.join(scores_dir, group)
        if not os.path.isdir(group_path):
            continue

        print(f"处理组: {group}")

        for task_id in os.listdir(group_path):
            task_path = os.path.join(group_path, task_id)
            if not os.path.isdir(task_path):
                continue

            for score_file in os.listdir(task_path):
                if score_file.endswith('.json'):
                    model_name = score_file.replace('.json', '')
                    file_path = os.path.join(task_path, score_file)
                    score_files.append((file_path, group, task_id, model_name))

    return score_files



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='处理模型评测结果并存入MongoDB')
    parser.add_argument('--scores_dir', type=str,
                        default='/mnt/e/Results/chemechpred/scores/_random313',
                        help='评测结果目录路径')
    parser.add_argument('--mongo_host', type=str, default='100.84.70.2',
                        help='MongoDB主机地址')
    parser.add_argument('--mongo_port', type=int, default=27017,
                        help='MongoDB端口')
    parser.add_argument('--db_name', type=str, default='chemechpred',
                        help='数据库名称')
    parser.add_argument('--collection_name', type=str, default='model_scores',
                        help='集合名称')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 连接MongoDB
    try:
        client = MongoClient(args.mongo_host, args.mongo_port)
        client.admin.command('ping')  # 测试连接
        db = client[args.db_name]

        if args.collection_name in db.list_collection_names():
            print(f"集合 {args.collection_name} 已存在，正在删除...")
            db[args.collection_name].drop()
            print(f"集合 {args.collection_name} 已删除")

        collection = db[args.collection_name]
        print(f"成功连接到MongoDB: {args.mongo_host}:{args.mongo_port}")
    except ConnectionFailure as e:
        print(f"连接MongoDB失败: {e}")
        return

    # 查找所有评分文件
    print(f"开始扫描目录: {args.scores_dir}")
    score_files = find_all_score_files(args.scores_dir)
    print(f"找到 {len(score_files)} 个评分文件")

    # 处理文件并存入数据库
    success_count = 0
    error_count = 0

    for file_path, group, task_id, model_name in tqdm(score_files, desc="处理文件"):
        try:
            # 处理单个文件
            processed_data = process_score_file(file_path, group, task_id, model_name)

            # 存入MongoDB
            collection.insert_one(processed_data)
            success_count += 1

        except Exception as e:
            print(f"处理文件失败 {file_path}: {e}")
            error_count += 1

    # 输出统计信息
    print(f"\n处理完成!")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"总计: {len(score_files)}")

    # 关闭连接
    client.close()


if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     process_score_file('results/chemechpred/scores/_random313/rxn_to_mech/updcanoamrxn_to_cls_mech/enhc_rxts_to_prds_v2_1.json', 'enhc_rxts_to_prds', 'updcanoamrxn_to_cls_mech', 'enhc_rxts_to_prds_v2_1')