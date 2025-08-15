# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: messy_code2.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/14 10:21

import json
from pathlib import Path

def try_decode_token(token):
    """
    尝试将 Latin-1 形式的 UTF-8 字节序列还原成可读字符串。
    如果失败，返回原 token。
    """
    try:
        # 检查是否只包含 Latin-1 范围内的字符
        if all(ord(ch) < 256 for ch in token):
            return token.encode("latin1").decode("utf-8")
        else:
            return token
    except UnicodeDecodeError:
        return token

def decode_tokenizer_vocab(input_path, output_path, preview_count=20):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    if not vocab:
        print("❌ 未找到 vocab 字段，请检查文件结构。")
        return

    fixed_vocab = {}
    changed_count = 0

    for token, idx in vocab.items():
        fixed_token = try_decode_token(token)
        if fixed_token != token:
            changed_count += 1
        fixed_vocab[fixed_token] = idx

    tokenizer_data["model"]["vocab"] = fixed_vocab

    # 保存新文件（可读版）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 解码完成！共还原 {changed_count} 个 token。")
    print(f"💾 已保存到: {output_path}")

    # 预览前 N 个结果
    print("\n📌 预览前 {} 个解码结果:".format(preview_count))
    for i, (k, v) in enumerate(fixed_vocab.items()):
        if i >= preview_count:
            break
        print(f"{i+1}. {repr(k)} -> ID {v}")

if __name__ == "__main__":
    # 修改为你的文件路径
    input_file = "/mnt/e/Models/Qwen/Qwen2-0.5B/tokenizer.json"
    output_file = "tokenizer_readable.json"
    decode_tokenizer_vocab(input_file, output_file)
