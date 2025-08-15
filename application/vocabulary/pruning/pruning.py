# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: pruning.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/15 9:29

import json
import os
import shutil
from pathlib import Path


def is_ascii(s):
    """检查字符串是否只包含ASCII字符"""
    return all(ord(c) < 128 for c in s)


def prune_vocab(vocab):
    """缩减词表，保留只包含ASCII字符的token"""
    pruned_vocab = {}
    for token, idx in vocab.items():
        if is_ascii(token):
            pruned_vocab[token] = idx
    return pruned_vocab


def prune_merges(merges):
    """缩减merges列表，保留只包含ASCII字符的merge"""
    pruned_merges = []
    for merge in merges:
        parts = merge.split()
        if all(is_ascii(part) for part in parts):
            pruned_merges.append(merge)
    return pruned_merges


def update_tokenizer_json(tokenizer_path, pruned_vocab, pruned_merges):
    """更新tokenizer.json文件"""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer = json.load(f)

    # 更新vocab
    tokenizer['model']['vocab'] = pruned_vocab

    # 更新merges
    tokenizer['model']['merges'] = pruned_merges

    # 更新added_tokens（如果有非ASCII的也需要移除）
    if 'added_tokens' in tokenizer:
        tokenizer['added_tokens'] = [
            token for token in tokenizer['added_tokens']
            if is_ascii(token['content'])
        ]

    return tokenizer


def main():
    # 原始模型路径
    model_path = "/home/liangtao/Models/Qwen/Qwen2-0.5B/"
    # 新模型路径
    new_model_path = "/home/liangtao/Models/Qwen/Qwen2-0.5B-VocabPruned/"

    # 创建新目录
    os.makedirs(new_model_path, exist_ok=True)

    # 1. 处理tokenizer.json
    tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # 获取原始vocab和merges
    original_vocab = tokenizer_data['model']['vocab']
    original_merges = tokenizer_data['model']['merges']

    # 缩减vocab和merges
    pruned_vocab = prune_vocab(original_vocab)
    pruned_merges = prune_merges(original_merges)

    # 更新tokenizer.json
    updated_tokenizer = update_tokenizer_json(tokenizer_json_path, pruned_vocab, pruned_merges)

    # 保存新的tokenizer.json
    new_tokenizer_path = os.path.join(new_model_path, "tokenizer.json")
    with open(new_tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(updated_tokenizer, f, ensure_ascii=False, indent=2)

    # 2. 处理vocab.json (如果有)
    vocab_json_path = os.path.join(model_path, "vocab.json")
    if os.path.exists(vocab_json_path):
        with open(vocab_json_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        pruned_vocab_json = {k: v for k, v in vocab_data.items() if is_ascii(k)}
        with open(os.path.join(new_model_path, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(pruned_vocab_json, f, ensure_ascii=False, indent=2)

    # 3. 处理merges.txt (如果有)
    merges_txt_path = os.path.join(model_path, "merges.txt")
    if os.path.exists(merges_txt_path):
        with open(merges_txt_path, 'r', encoding='utf-8') as f:
            merges_lines = f.readlines()
        pruned_merges_lines = [line for line in merges_lines if all(is_ascii(part) for part in line.strip().split())]
        with open(os.path.join(new_model_path, "merges.txt"), 'w', encoding='utf-8') as f:
            f.writelines(pruned_merges_lines)

    # 4. 复制其他不需要修改的文件
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors"
    ]

    for file in files_to_copy:
        src = os.path.join(model_path, file)
        dst = os.path.join(new_model_path, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print(f"词表缩减完成，新模型已保存到: {new_model_path}")


if __name__ == "__main__":
    main()