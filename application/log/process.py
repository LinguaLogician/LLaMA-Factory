# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: process.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/8 19:01
# https://chat.deepseek.com/a/chat/s/f24b1652-c17e-4abe-bb46-780ac8abcf47

import os
import re
import argparse
from pathlib import Path
import json
from tqdm import tqdm


def process_log_files(base_dir, output_dir, pattern, files):
    """
    处理日志文件，去除匹配正则表达式的行和空行

    Args:
        base_dir (str): 输入日志文件目录
        output_dir (str): 输出目录
        pattern (str): 要去除行的正则表达式模式
        files (list or str): 要处理的文件列表或'all'
    """
    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取要处理的文件列表
    if files == 'all':
        log_files = [f for f in os.listdir(base_dir) if f.endswith('.log')]
    else:
        # 确保传入的是文件列表
        if isinstance(files, str):
            try:
                files = json.loads(files.replace("'", '"'))  # 将单引号转换为双引号以便JSON解析
            except json.JSONDecodeError:
                files = [files]  # 如果无法解析为JSON，假设是单个文件名
        log_files = [f for f in files if f.endswith('.log')]

    if not log_files:
        print(f"在目录 {base_dir} 中没有找到要处理的.log文件")
        return

    print(f"找到 {len(log_files)} 个日志文件需要处理")

    # 编译正则表达式
    regex = re.compile(pattern)

    processed_count = 0
    # 使用tqdm创建进度条
    for filename in tqdm(log_files, desc="处理日志文件", unit="file"):
        input_path = os.path.join(base_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            tqdm.write(f"警告: 文件 {filename} 不存在，跳过处理")
            continue

        try:
            with open(input_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            original_line_count = len(lines)

            # 过滤掉匹配正则表达式的行和空行
            filtered_lines = []
            for line in lines:
                # 去除匹配正则表达式的行
                if regex.match(line):
                    continue
                # 去除空行（包括只包含空白字符的行）
                if line.strip() == '':
                    continue
                filtered_lines.append(line)

            # 写入处理后的文件
            with open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.writelines(filtered_lines)

            tqdm.write(f"✓ {filename}: {original_line_count} → {len(filtered_lines)} 行 "
                       f"(移除 {original_line_count - len(filtered_lines)} 行)")
            processed_count += 1

        except UnicodeDecodeError:
            try:
                # 尝试使用其他编码
                with open(input_path, 'r', encoding='latin-1') as infile:
                    lines = infile.readlines()

                original_line_count = len(lines)
                filtered_lines = []
                for line in lines:
                    if regex.match(line):
                        continue
                    if line.strip() == '':
                        continue
                    filtered_lines.append(line)

                with open(output_path, 'w', encoding='utf-8') as outfile:
                    outfile.writelines(filtered_lines)

                tqdm.write(f"✓ {filename} (latin-1编码): {original_line_count} → {len(filtered_lines)} 行")
                processed_count += 1

            except Exception as e:
                tqdm.write(f"✗ 处理文件 {filename} 时出错: {e}")

        except Exception as e:
            tqdm.write(f"✗ 处理文件 {filename} 时出错: {e}")

    print(f"\n处理完成! 成功处理 {processed_count}/{len(log_files)} 个文件")


def main():
    parser = argparse.ArgumentParser(description='处理日志文件，去除匹配特定模式的行和空行')
    parser.add_argument('--base_dir', type=str, default='/mnt/e/Development/LLMSpace/LLaMA-Factory/logs',
                        help='输入日志文件目录')
    parser.add_argument('--output_dir', type=str, default='/mnt/e/Development/LLMSpace/LLaMA-Factory/logs/processed',
                        help='输出目录')
    parser.add_argument('--pattern', type=str, default=r'^.*[0-9]%\|.*\n',
                        help='要去除行的正则表达式模式')
    parser.add_argument('--files', type=str, default='all',
                        help='要处理的文件列表，如：\'["file1.log","file2.log"]\' 或 "all"')

    args = parser.parse_args()

    print(f"开始处理日志文件...")
    print(f"输入目录: {args.base_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"过滤模式: {args.pattern}")
    print(f"处理文件: {args.files}")
    print("-" * 50)

    process_log_files(args.base_dir, args.output_dir, args.pattern, args.files)


if __name__ == '__main__':
    main()