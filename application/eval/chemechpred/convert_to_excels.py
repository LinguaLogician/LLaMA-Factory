# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: convert_to_excels.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/1 15:43
# -*- coding: utf-8 -*-
# https://chat.deepseek.com/a/chat/s/28c1672d-8ba4-452a-bdb7-876623fe203f
import os
import json
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, List, Any
import argparse
from tqdm import tqdm


def load_fields_config(config_path: str) -> Dict[str, Any]:
    """加载字段配置"""
    if not os.path.exists(config_path):
        # 创建默认配置
        default_config = {
            "basic_info": {
                "group": "组别",
                "task": "任务",
                "model": "模型"
            },
            "mols": {
                "accTopK": "AccTopK",
                "accKth": "AccKth",
                "accTopKMols": "AccTopKMols",
                "accKthMols": "AccKthMols",
                "precTopKFormat": "格式正确率",
                "precKthFormat": "格式正确率(Kth)",
                "precTopKManner": "方式正确率",
                "precKthManner": "方式正确率(Kth)",
                "precTopKValid": "有效正确率",
                "precKthValid": "有效正确率(Kth)"
            },
            "cls": {
                "accTopK": "AccTopK",
                "accKth": "AccKth"
            },
            "mech": {
                "accTopK": "AccTopK",
                "accKth": "AccKth"
            },
            "overall": {
                "matched": "匹配正确率",
                "resolved": "解析正确率",
                "totally": "总体正确率"
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"已创建默认字段配置文件: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_value_by_path(data: Dict, path: str, k_values: List[str]) -> Dict[str, Any]:
    """根据路径提取值，处理K值"""
    keys = path.split('.')
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return {f"k_{k}": None for k in k_values}

    # 如果是字典且包含K值，过滤出指定的K值
    if isinstance(current, dict):
        return {f"k_{k}": current.get(f"k_{k}") for k in k_values}
    else:
        return {f"k_{k}": current for k in k_values}


def build_excel_data(mongo_data: List[Dict], fields_config: Dict, k_values: List[str]) -> pd.DataFrame:
    """构建Excel数据"""
    all_rows = []

    for item in tqdm(mongo_data, desc="处理数据"):
        row_data = {}

        # 基础信息
        for field, display_name in fields_config['basic_info'].items():
            row_data[display_name] = item.get(field, '')

        # 处理各个部分的指标
        sections = ['mols', 'cls', 'mech', 'overall']

        for section in sections:
            if section not in fields_config:
                continue

            for metric, display_name in fields_config[section].items():
                if section == 'overall':
                    # 总体指标在根级别
                    path = f"{metric}.precTopK"
                    k_data = extract_value_by_path(item, path, k_values)
                else:
                    # 预测项指标在对应section下
                    path = f"{section}.{metric}"
                    k_data = extract_value_by_path(item, path, k_values)

                for k, value in k_data.items():
                    col_name = f"{display_name}_{k}"
                    row_data[col_name] = value

        all_rows.append(row_data)

    return pd.DataFrame(all_rows)


def create_merged_headers(df: pd.DataFrame, fields_config: Dict, k_values: List[str]) -> List:
    """创建合并表头"""
    # 第一行：基础信息 + 各个部分的合并表头
    header_row1 = []
    header_row2 = []

    # 基础信息列
    basic_cols = list(fields_config['basic_info'].values())
    header_row1.extend([('基础信息', len(basic_cols))] + [''] * (len(basic_cols) - 1))
    header_row2.extend(basic_cols)

    # 各个部分的列
    sections = ['mols', 'cls', 'mech', 'overall']

    for section in sections:
        if section not in fields_config:
            continue

        section_cols = []
        for metric, display_name in fields_config[section].items():
            for k in k_values:
                section_cols.append(f"{display_name}_k_{k}")

        if section_cols:
            section_name = {
                'mols': '分子预测',
                'cls': '分类预测',
                'mech': '机制预测',
                'overall': '总体指标'
            }.get(section, section)

            header_row1.append((section_name, len(section_cols)))
            header_row1.extend([''] * (len(section_cols) - 1))
            header_row2.extend(section_cols)

    return [header_row1, header_row2]


def save_to_excel_with_merged_headers(df: pd.DataFrame, output_path: str, fields_config: Dict, k_values: List[str]):
    """保存到Excel并设置合并表头"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 创建工作表
        df.to_excel(writer, sheet_name='评测结果', index=False, startrow=2)

        workbook = writer.book
        worksheet = writer.sheets['评测结果']

        # 创建合并表头
        headers = create_merged_headers(df, fields_config, k_values)

        # 写入第一行表头（合并单元格）
        for col_idx, header_info in enumerate(headers[0], 1):
            if isinstance(header_info, tuple):
                header_name, col_span = header_info
                if col_span > 1:
                    worksheet.merge_cells(
                        start_row=1, start_column=col_idx,
                        end_row=1, end_column=col_idx + col_span - 1
                    )
                worksheet.cell(row=1, column=col_idx, value=header_name)
            elif header_info:
                worksheet.cell(row=1, column=col_idx, value=header_info)

        # 写入第二行表头（具体列名）
        for col_idx, col_name in enumerate(headers[1], 1):
            worksheet.cell(row=2, column=col_idx, value=col_name)

        # 设置列宽
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            adjusted_width = min(max_length + 2, 20)
            worksheet.column_dimensions[column_letter].width = adjusted_width


def main():
    """主函数"""
    args = parse_args()

    # 解析参数
    k_values = [k.strip() for k in args.k_values.split(',')]

    try:
        query = json.loads(args.query)
    except json.JSONDecodeError:
        print("查询条件格式错误，使用默认查询条件")
        query = {}

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载字段配置
    print("加载字段配置...")
    fields_config = load_fields_config(args.fields_config)

    # 连接MongoDB
    try:
        client = MongoClient(args.mongo_host, args.mongo_port)
        db = client[args.db_name]
        collection = db[args.collection_name]
        print(f"成功连接到MongoDB: {args.mongo_host}:{args.mongo_port}")
    except Exception as e:
        print(f"连接MongoDB失败: {e}")
        return

    # 查询数据
    print("查询数据...")
    mongo_data = list(collection.find(query))
    print(f"找到 {len(mongo_data)} 条记录")

    if not mongo_data:
        print("没有找到数据，退出")
        return

    # 构建Excel数据
    print("构建Excel数据...")
    df = build_excel_data(mongo_data, fields_config, k_values)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"chemechpred_{timestamp}.xlsx"
    output_path = os.path.join(args.output_dir, output_filename)

    # 保存到Excel
    print("保存到Excel...")
    save_to_excel_with_merged_headers(df, output_path, fields_config, k_values)

    print(f"Excel文件已保存: {output_path}")
    print(f"包含 {len(df)} 行数据")

    # 关闭连接
    client.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从MongoDB查询数据并生成Excel文件')
    parser.add_argument('--mongo_host', type=str, default='100.84.70.2',
                        help='MongoDB主机地址')
    parser.add_argument('--mongo_port', type=int, default=27017,
                        help='MongoDB端口')
    parser.add_argument('--db_name', type=str, default='chemechpred',
                        help='数据库名称')
    parser.add_argument('--collection_name', type=str, default='model_scores',
                        help='集合名称')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/e/Results/chemechpred/excels/',
                        help='输出目录路径')
    parser.add_argument('--fields_config', type=str,
                        default='excel_fields_config.json',
                        help='字段配置JSON文件路径')
    parser.add_argument('--k_values', type=str, default='1,2,3,4,5',
                        help='要包含的K值，用逗号分隔')
    parser.add_argument('--query', type=str, default='{}',
                        help='MongoDB查询条件，JSON格式字符串')
    return parser.parse_args()

if __name__ == '__main__':
    main()