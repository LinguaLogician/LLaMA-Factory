# https://chat.deepseek.com/a/chat/s/5fc6b24d-5cb3-450e-88e5-2a9fc104baad

# -*- coding: utf-8 -*-
import os
import json
import re
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd
from tqdm import tqdm
import math


class MongoToExcelExporter:
    def __init__(self, mongo_host: str, mongo_port: int, db_name: str, collection_name: str):
        self.mongo_host = mongo_host
        self.mongo_port = mongo_port
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def connect(self) -> bool:
        """连接MongoDB"""
        try:
            self.client = MongoClient(self.mongo_host, self.mongo_port)
            self.client.admin.command('ping')
            self.collection = self.client[self.db_name][self.collection_name]
            print(f"成功连接到MongoDB: {self.mongo_host}:{self.mongo_port}")
            return True
        except ConnectionFailure as e:
            print(f"连接MongoDB失败: {e}")
            return False

    def disconnect(self):
        """断开MongoDB连接"""
        if self.client:
            self.client.close()

    def build_query(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """构建MongoDB查询条件"""
        query = {}

        # 处理精确匹配条件
        filters = config.get('query_filters', {})
        for field in ['group', 'task', 'model']:
            if field in filters and filters[field]:
                query[field] = {'$in': filters[field]}

        # 处理正则匹配条件
        patterns = config.get('query_patterns', {})
        for field, pattern in patterns.items():
            if pattern:
                if field in query:
                    # 如果已有精确匹配条件，添加正则条件
                    query[field]['$regex'] = pattern
                else:
                    query[field] = {'$regex': pattern}

        return query

    def extract_field_value(self, data: Dict, field_path: str) -> Any:
        """使用JSONPath-like语法提取字段值"""
        if not field_path:
            return None

        keys = field_path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def process_value(self, value: Any, config: Dict[str, Any]) -> Any:
        """处理数值：乘以100、保留小数位等"""
        if isinstance(value, (int, float)) and not math.isnan(value):
            if config.get('multiply_by_100', False):
                value = value * 100

            decimal_places = config.get('decimal_places', 2)
            if decimal_places >= 0:
                value = round(value, decimal_places)

        return value

    def build_excel_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """构建Excel数据"""
        print("构建查询条件...")
        query = self.build_query(config)

        print("执行数据库查询...")
        cursor = self.collection.find(query)
        total_docs = self.collection.count_documents(query)

        if total_docs == 0:
            print("未找到匹配的文档")
            return []

        excel_data = []
        export_fields = config.get('export_fields', {})
        k_values = config.get('k_values_to_export', [])

        print("处理数据...")
        for doc in tqdm(cursor, total=total_docs, desc="处理文档"):
            row_data = {}

            # 处理基础信息
            basic_info = export_fields.get('basic_info', {})
            for field, alias in basic_info.items():
                row_data[alias] = doc.get(field, '')

            # 处理各个预测类型的指标
            for pred_type in ['cls', 'mech', 'mols', 'matched', 'resolved', 'totally']:
                if pred_type in export_fields:
                    pred_config = export_fields[pred_type]
                    pred_data = doc.get(pred_type, {})

                    for metric, metric_alias in pred_config.items():
                        metric_data = pred_data.get(metric, {})

                        for k_value in k_values:
                            if k_value in metric_data:
                                field_name = f"{pred_type}_{metric}_{k_value}"
                                display_name = f"{metric_alias} ({config['k_value_aliases'].get(k_value, k_value)})"
                                value = self.process_value(metric_data[k_value], config)
                                row_data[display_name] = value
                            else:
                                # 如果该k值不存在，填充空值
                                field_name = f"{pred_type}_{metric}_{k_value}"
                                display_name = f"{metric_alias} ({config['k_value_aliases'].get(k_value, k_value)})"
                                row_data[display_name] = ''

            excel_data.append(row_data)

        return excel_data

    def create_excel_with_headers(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """创建带有多级表头的Excel数据"""
        if df.empty:
            return df

        # 获取所有列名
        columns = list(df.columns)

        # 基础信息列
        basic_columns = []
        for col in columns:
            if not any(x in col for x in ['_k_1', '_k_2', '_k_3', '_k_4', '_k_5']):
                basic_columns.append(col)

        # 指标列
        metric_columns = [col for col in columns if col not in basic_columns]

        # 重新组织列顺序
        export_fields = config.get('export_fields', {})
        k_values = config.get('k_values_to_export', [])
        k_aliases = config.get('k_value_aliases', {})

        # new_columns = basic_columns.copy()
        new_columns = []
        header_rows = []
        first_row, second_row, third_row = [], [], []

        basic_info = export_fields.get('basic_info', {})
        for field, alias in basic_info.items():
            new_columns.append(alias)
            # 第一行表头：基础信息 + 预测类型
            first_row.append('')
            # 第二行表头：基础信息 + 指标名称
            second_row.append('')
            # 第三行表头：基础信息 + k值
            third_row.append(alias)

        # 按预测类型组织指标
        for pred_type in ['cls', 'mech', 'mols', 'matched', 'resolved', 'totally']:
            if pred_type in export_fields:
                pred_config = export_fields[pred_type]

                for metric, metric_alias in pred_config.items():
                    # 为每个k值创建列
                    for k_value in k_values:
                        col_name = f"{metric_alias} ({k_aliases.get(k_value, k_value)})"
                        if col_name in df.columns:
                            new_columns.append(col_name)

                            # 填充表头信息
                            first_row.append(pred_type.upper())
                            second_row.append(metric_alias)
                            third_row.append(k_aliases.get(k_value, k_value))

        # 重新排列DataFrame
        df_reordered = df[new_columns]

        # 创建多级索引
        arrays = [first_row, second_row, third_row]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['Prediction Type', 'Metric', 'K Value'])
        df_reordered.columns = index

        return df_reordered

    # convert_to_excels2.py

    # 在 export_to_excel 方法中，修改这一行：
    def export_to_excel(self, config: Dict[str, Any], output_file: str):
        """导出数据到Excel"""
        print("开始导出数据...")

        # 构建数据
        excel_data = self.build_excel_data(config)

        if not excel_data:
            print("没有数据可导出")
            return False

        # 创建DataFrame
        df = pd.DataFrame(excel_data)

        # 创建带有多级表头的DataFrame
        df_with_headers = self.create_excel_with_headers(df, config)

        # 导出到Excel
        print(f"导出到Excel文件: {output_file}")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 修改这一行：去掉 index=False 或者改为 index=True
            df_with_headers.to_excel(writer, sheet_name='Model Scores', index=True)

            # 获取工作表并设置格式
            worksheet = writer.sheets['Model Scores']

            # 设置列宽
            # for column in worksheet.columns:
            # 在 export_to_excel 方法中，替换设置列宽的部分：
            export_fields = config.get('export_fields', {})

            k_values_to_export = config.get('k_values_to_export', [])
            basic_info = export_fields.get('basic_info', {})
            for i, column in enumerate(df_with_headers.columns, 1):
                max_length = 0
                column_letter = worksheet.cell(row=3, column=i).column_letter
                basic_info_alias = None
                # 检查表头长度（多级表头有3行）
                for row in range(1, 4):  # 遍历3行表头
                    cell_value = worksheet.cell(row=row, column=i).value
                    if cell_value:
                        if cell_value in basic_info.values():
                            basic_info_alias = cell_value

                        header_length = len(str(cell_value))
                        if header_length > max_length:
                            max_length = header_length

                # 检查数据单元格长度
                for row in range(5, len(df_with_headers) + 5):  # 数据从第5行开始（第4行是列名）
                    cell_value = worksheet.cell(row=row, column=i).value
                    if cell_value:
                        cell_length = len(str(cell_value))
                        if cell_length > max_length:
                            max_length = cell_length

                # 设置列宽，限制最大宽度
                if basic_info_alias:
                    adjusted_width = max_length + 2
                else:
                    adjusted_width = 8
                worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"成功导出 {len(excel_data)} 行数据到 {output_file}")
        return True


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从MongoDB导出数据到Excel')
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
                        help='输出目录')
    parser.add_argument('--config_path', type=str,
                        default='application/chemech/eval/_config/excel_query/',
                        help='配置文件目录')
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG_FILE,
                        help='配置文件名(不含路径)')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 构建完整配置文件路径
    config_file_path = os.path.join(args.config_path, args.config_file)
    if not config_file_path.endswith('.json'):
        config_file_path += '.json'

    # 加载配置
    config = load_config(config_file_path)
    if not config:
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成输出文件名
    config_base_name = os.path.splitext(args.config_file)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"{config_base_name}_{timestamp}.xlsx")

    # 创建导出器并执行导出
    exporter = MongoToExcelExporter(
        mongo_host=args.mongo_host,
        mongo_port=args.mongo_port,
        db_name=args.db_name,
        collection_name=args.collection_name
    )

    try:
        if exporter.connect():
            success = exporter.export_to_excel(config, output_file)
            if success:
                print("导出完成!")
            else:
                print("导出失败!")
    finally:
        exporter.disconnect()


if __name__ == '__main__':
    # DEFAULT_CONFIG_FILE = 'atom_mapping.json'
    # DEFAULT_CONFIG_FILE = 'multi_enhc_rxts_to_prds_v1_1.json'
    # DEFAULT_CONFIG_FILE = 'multi_enhc_rxts_to_prds_v7_1.json'
    DEFAULT_CONFIG_FILE = 'multi_enhc_prds_to_rxts_v1_1.json'
    # DEFAULT_CONFIG_FILE = 'single_reaction_classification.json'
    main()