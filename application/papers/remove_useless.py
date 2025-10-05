# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: remove_useless.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/10/3 16:40
# https://chat.deepseek.com/a/chat/s/d1ad17df-67e9-4da6-a03b-75a7bb52167b
# !/usr/bin/env python3
"""
Overleaf项目参考文献处理工具
用于提取tex文件中引用的文献并生成新的bib文件
"""

import os
import re
import argparse
from pathlib import Path
from typing import Set, List, Dict
from tqdm import tqdm


class BibProcessor:
    """参考文献处理器"""

    def __init__(self, chapters_dir: str, ref_dir: str, output_filename: str = "new_ref.bib"):
        """
        初始化处理器

        Args:
            chapters_dir: 章节文件目录
            ref_dir: 参考文献目录
            output_filename: 输出文件名
        """
        self.chapters_dir = Path(chapters_dir)
        self.ref_dir = Path(ref_dir)
        self.output_file = self.ref_dir / output_filename

    def extract_cite_keys_from_tex(self, tex_file: Path) -> Set[str]:
        """
        从tex文件中提取所有引用标签

        Args:
            tex_file: tex文件路径

        Returns:
            引用标签集合
        """
        cite_keys = set()
        cite_pattern = r'\\cite\{([^}]+)\}'

        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()

            matches = re.findall(cite_pattern, content)
            for match in matches:
                # 处理多个引用标签用逗号分隔的情况
                keys = [key.strip() for key in match.split(',')]
                cite_keys.update(keys)

        except Exception as e:
            print(f"错误: 读取文件 {tex_file} 时出错: {e}")

        return cite_keys

    def parse_bib_file(self, bib_file: Path) -> Dict[str, str]:
        """
        解析bib文件，提取所有参考文献条目

        Args:
            bib_file: bib文件路径

        Returns:
            字典: {引用标签: 完整bib条目}
        """
        bib_entries = {}

        try:
            with open(bib_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 改进的bib条目解析方法
            entries = self._split_bib_entries(content)

            for entry in entries:
                cite_key = self._extract_cite_key(entry)
                if cite_key:
                    bib_entries[cite_key] = entry

        except Exception as e:
            print(f"错误: 解析bib文件 {bib_file} 时出错: {e}")

        return bib_entries

    def _split_bib_entries(self, content: str) -> List[str]:
        """
        将bib文件内容分割成单独的条目

        Args:
            content: bib文件内容

        Returns:
            条目列表
        """
        entries = []
        current_entry = []
        brace_level = 0
        in_entry = False

        for line in content.split('\n'):
            stripped_line = line.strip()

            # 检测条目开始
            if stripped_line.startswith('@') and not in_entry:
                in_entry = True
                current_entry = [line]
                # 计算第一行的括号层级
                brace_level += self._count_braces(line)
                continue

            if in_entry:
                current_entry.append(line)
                brace_level += self._count_braces(line)

                # 当括号层级归零时，条目结束
                if brace_level == 0:
                    entries.append('\n'.join(current_entry))
                    current_entry = []
                    in_entry = False

        return entries

    def _count_braces(self, line: str) -> int:
        """
        计算一行中括号的层级变化

        Args:
            line: 文本行

        Returns:
            括号层级变化
        """
        count = 0
        in_quotes = False
        escaped = False

        for char in line:
            if escaped:
                escaped = False
                continue

            if char == '\\':
                escaped = True
                continue

            if char == '"' and not escaped:
                in_quotes = not in_quotes
                continue

            if not in_quotes:
                if char == '{':
                    count += 1
                elif char == '}':
                    count -= 1

        return count

    def _extract_cite_key(self, entry: str) -> str:
        """
        从bib条目中提取引用标签

        Args:
            entry: bib条目内容

        Returns:
            引用标签
        """
        # 匹配 @type{key, 的模式
        pattern = r'@\w+\{([^,]+),'
        match = re.search(pattern, entry)
        if match:
            return match.group(1).strip()
        return ""

    def get_all_tex_files(self) -> List[Path]:
        """
        获取所有tex文件

        Returns:
            tex文件路径列表
        """
        tex_files = list(self.chapters_dir.glob("chapter-*.tex"))
        tex_files.sort()
        return tex_files

    def process_references(self) -> bool:
        """
        处理参考文献的主要流程

        Returns:
            处理是否成功
        """
        print("开始处理参考文献...")

        # 1. 获取所有tex文件
        tex_files = self.get_all_tex_files()
        if not tex_files:
            print(f"错误: 在目录 {self.chapters_dir} 中未找到chapter-*.tex文件")
            return False

        print(f"找到 {len(tex_files)} 个tex文件")

        # 2. 提取所有引用标签
        print("提取引用标签...")
        all_cite_keys = set()

        for tex_file in tqdm(tex_files, desc="扫描tex文件"):
            cite_keys = self.extract_cite_keys_from_tex(tex_file)
            all_cite_keys.update(cite_keys)
            print(f"  {tex_file.name}: 找到 {len(cite_keys)} 个引用")

        print(f"总共找到 {len(all_cite_keys)} 个唯一的引用标签")
        print("引用标签示例:", list(all_cite_keys)[:5])

        # 3. 解析原始bib文件
        original_bib = self.ref_dir / "paper.bib"
        if not original_bib.exists():
            print(f"错误: 未找到bib文件 {original_bib}")
            return False

        print("解析原始bib文件...")
        bib_entries = self.parse_bib_file(original_bib)
        print(f"原始bib文件中包含 {len(bib_entries)} 个参考文献条目")
        print("bib条目标签示例:", list(bib_entries.keys())[:5])

        # 4. 筛选被引用的文献
        print("筛选被引用的文献...")
        cited_entries = {}
        missing_keys = []

        for key in tqdm(all_cite_keys, desc="筛选文献"):
            if key in bib_entries:
                cited_entries[key] = bib_entries[key]
            else:
                missing_keys.append(key)

        # 5. 输出缺失的引用标签
        if missing_keys:
            print(f"警告: 以下 {len(missing_keys)} 个引用标签在bib文件中未找到:")
            for key in missing_keys:
                print(f"  - {key}")

        # 6. 写入新的bib文件
        print(f"写入新的bib文件: {self.output_file}")
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for key, entry in tqdm(cited_entries.items(), desc="写入文件"):
                    f.write(entry + '\n\n')

            print(f"成功! 生成了包含 {len(cited_entries)} 个参考文献条目的新bib文件")
            return True

        except Exception as e:
            print(f"错误: 写入文件 {self.output_file} 时出错: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='处理Overleaf项目参考文献')
    parser.add_argument('--chapters_dir', type=str,
                        default='/mnt/e/PaperWorks/Graduation/chapters',
                        help='章节文件目录路径')
    parser.add_argument('--ref_dir', type=str,
                        default='/mnt/e/PaperWorks/Graduation/references',
                        help='参考文献目录路径')
    parser.add_argument('--output', type=str,
                        default='new_ref.bib',
                        help='输出bib文件名')

    args = parser.parse_args()

    # 创建处理器并执行
    processor = BibProcessor(
        chapters_dir=args.chapters_dir,
        ref_dir=args.ref_dir,
        output_filename=args.output
    )

    success = processor.process_references()

    if success:
        print("参考文献处理完成!")
    else:
        print("参考文献处理失败!")
        exit(1)


if __name__ == "__main__":
    main()