# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: sampling.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/6/3 22:06

'''
请为我编写Python程序
已知：
1. base_dir=/home/liangtao/DataSets/Chemistry/MolecularTransformer/
2. base_dir下有目录space和nospace, 它们之下都有数据集test文件夹
3. test文件夹下都有一系列相关数据文件，格式为json，文件名为f"{file_name}.json"，表示相关的数据集
4. 每一个json文件的内容形如：
[
  {
    "id": "MIT_mixed_test_space_000000001",
    "instruction": "PREDICT_PRODUCT:",
    "input": "C 1 C C O C 1 . N # C c 1 c c s c 1 N . O = [N+] ( [O-] ) c 1 c c ( F ) c ( F ) c c 1 F . [H-] . [Na+]",
    "output": "N # C c 1 c c s c 1 N c 1 c c ( F ) c ( F ) c c 1 [N+] ( = O ) [O-]"
  },
  ...
]
要求：
1. 按上述要求获得每一个test文件夹下的文件，为每个数据集json文件生成响应的子集
2. 创建自己的方式为，获得json文件数据后，随机采样一定数量的数据，采样的数据量可以自定义
3. 一次采样过程中，可以采样多个数量大小的数据集，例如一次性为每个json文件采样100、200、500、1000长度的子集, 得到的子集文件中的内容按照id从小到大排序
4. 采样得到的子集数据文件存放到对应test文件架下的f"subset{subset_size}_{style}"文件夹下，例如：subset100_random, subset200_random, subset500_random, subset1k_random
5. 对应的采样自己的文件夹要保存一份采样报告，txt格式，记录这个文件夹下所有文件的原来数据总量，最后来采样的自己的数据量，采样方式，采样时间等信息
6. 处理过程中，显示进度条
7. 方法进行适度封装，不要封装过度，从__main__开始执行
8. 相关参数通过arg parser处理，代码位于__main__中
9. 代码中的日志和注释不要用中文，代码意外其他部分可用中文表达
'''

import os
import json
import random
import argparse
from datetime import datetime
from tqdm import tqdm


def load_json_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_data(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def create_subset(data, subset_size, style='random'):
    """
    Create a subset of the data with specified size and sampling style.

    Args:
        data: List of data items
        subset_size: Size of the subset to create
        style: Sampling style ('random' for random sampling)

    Returns:
        Sorted subset of data
    """
    if style == 'random':
        subset = random.sample(data, min(subset_size, len(data)))
    else:
        raise ValueError(f"Unknown sampling style: {style}")

    # Sort by id
    return sorted(subset, key=lambda x: x['id'])


def generate_report(file_reports, subset_size, style, output_dir):
    """Generate a report file for the subset creation with all original file info."""
    report_content = f"""
    Subset Creation Report
    =====================
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Sampling style: {style}
    Subset size: {subset_size}
    Output directory: {output_dir}

    Original Dataset Information:
    """

    for file_report in file_reports:
        report_content += f"""
        File: {file_report['filename']}
        Original size: {file_report['original_size']}
        Subset size: {file_report['subset_size']}
        """

    report_path = os.path.join(output_dir, 'subset_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_content)


def format_size(size):
    """Format size for folder naming (e.g., 1000->1k, 5000->5k, 5100->5.1k)"""
    if size < 1000:
        return str(size)
    elif size % 1000 == 0:
        return f"{size // 1000}k"
    else:
        return f"{size / 1000:.1f}k".replace('.0k', 'k')


def process_dataset(base_dir, style, subset_sizes):
    """
    Process all JSON files in test folders under base_dir/space and base_dir/nospace.
    """

    for style_dir in styles:
        test_dir = os.path.join(base_dir, style_dir, 'test')
        if not os.path.exists(test_dir):
            continue

        json_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]

        for size in subset_sizes:
            # Create output directory for this subset size
            size_str = format_size(size)
            output_dir = os.path.join(test_dir, f"{style}{size_str}")
            os.makedirs(output_dir, exist_ok=True)

            file_reports = []

            for json_file in tqdm(json_files, desc=f"Processing {style_dir} files for size {size}"):
                file_path = os.path.join(test_dir, json_file)
                file_name = os.path.splitext(json_file)[0]
                data = load_json_data(file_path)

                subset = create_subset(data, size, style)
                output_file = os.path.join(output_dir, f"{file_name}.json")
                save_json_data(subset, output_file)

                file_reports.append({
                    'filename': json_file,
                    'original_size': len(data),
                    'subset_size': len(subset)
                })

            # Generate report after processing all files for this subset size
            generate_report(file_reports, size, style, output_dir)


def parse_size_argument(size_str):
    """Parse size argument from string like '100,200,500,1k'"""
    sizes = []
    for s in size_str.split(','):
        if s.endswith('k'):
            sizes.append(int(s[:-1]) * 1000)
        else:
            sizes.append(int(s))
    return sizes


if __name__ == '__main__':
    styles = ['space', 'nospace']
    parser = argparse.ArgumentParser(description='Create subsets of chemistry datasets.')
    parser.add_argument('--base_dir', type=str, default='/home/liangtao/DataSets/Chemistry/MolecularTransformer/',
                        help='Base directory containing space and nospace folders')
    parser.add_argument('--style', type=str, default='random',
                        help='Sampling style (currently only "random" supported)')
    parser.add_argument('--sizes', type=str, default='1k',
                        help='Comma-separated list of subset sizes (e.g., "100,200,500,1k")')
    args = parser.parse_args()

    subset_sizes = parse_size_argument(args.sizes)

    print(f"Processing datasets in: {args.base_dir}")
    print(f"Sampling style: {args.style}")
    print(f"Creating subsets of sizes: {subset_sizes}")

    process_dataset(args.base_dir, args.style, subset_sizes)

    print("Subset creation completed.")