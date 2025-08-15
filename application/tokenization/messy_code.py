# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: messy_code.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/14 9:58

import json

# 1. 读取原始的 tokenizer.json 文件
with open("/mnt/e/Models/Qwen/Qwen2-0.5B/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

# 2. 重新写入，设置 ensure_ascii=False 避免 Unicode 转义
with open("tokenizer_fixed.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)  # indent 可选，美化格式

print("处理完成，已生成 tokenizer_fixed.json")
