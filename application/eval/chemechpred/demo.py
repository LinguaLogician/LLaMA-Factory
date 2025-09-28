# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: demo.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/27 11:26
import re
from typing import Dict, Any, List


def parse_output_text(text: str, expected_output_tags: List[str]) -> Dict[str, Any]:
    """
    解析模型输出的文本，提取各个信息模块

    Returns:
        Dict containing:
            - parsed_results: 解析出的各个信息模块 {tag: text}
            - is_resolved_correct: 解析过程是否正确
            - is_task_matched: 是否匹配任务要求
            - missing_tags: 缺失的标签
            - extra_tags: 多余的标签
    """
    # 使用正则表达式匹配信息标识符（以冒号结尾的单词）
    pattern = r'([A-Za-z_\.]+:)\s*\n'
    matches = list(re.finditer(pattern, text))

    parsed_results = {}
    sections = []

    # 提取各个信息段
    for i, match in enumerate(matches):
        tag = match.group(1).rstrip(':')
        start_pos = match.end()

        # 查找下一个标签的位置或文本结束
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        content = text[start_pos:end_pos].strip()
        sections.append((tag, content))

    # 如果没有找到标签，尝试按行分割
    if not sections:
        lines = text.strip().split('\n')
        current_tag = None
        current_content = []

        for line in lines:
            if line.endswith(':'):
                if current_tag and current_content:
                    sections.append((current_tag, '\n'.join(current_content)))
                current_tag = line.rstrip(':')
                current_content = []
            else:
                current_content.append(line)

        if current_tag and current_content:
            sections.append((current_tag, '\n'.join(current_content)))

    # 转换为字典
    for tag, content in sections:
        parsed_results[tag.lower().replace('.', '_').replace('->', '_to_')] = content

    # 检查解析正确性
    is_resolved_correct = len(sections) > 0

    # 检查任务匹配度
    expected_lower = [tag.lower().replace('.', '_').replace('->', '_to_') for tag in expected_output_tags]
    found_tags = list(parsed_results.keys())

    missing_tags = [tag for tag in expected_lower if tag not in found_tags]
    extra_tags = [tag for tag in found_tags if tag not in expected_lower]
    is_task_matched = len(missing_tags) == 0

    return {
        "parsed_results": parsed_results,
        "is_resolved_correct": is_resolved_correct,
        "is_task_matched": is_task_matched,
        "missing_tags": missing_tags,
        "extra_tags": extra_tags
    }

if __name__ == "__main__":

    predicted_text="UPD.CANO.STD.PRDS:\nCCOC(=O)C(=O)NN"
    output_tags = ['UPD.CANO.STD.PRDS']
    parsed_pred = parse_output_text(predicted_text, output_tags)
    print(parsed_pred)