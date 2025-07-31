# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: sparse_cache.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/7/30 20:37

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name_or_path="/home/liangtao/Models/Qwen/Qwen2-7B"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


class SparseCacheModel:
    def __init__(self, model, window_size=512):
        self.model = model
        self.window_size = window_size  # 滑动窗口大小

    def generate(self, input_text, max_length=50):
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        past_key_values = None

        for _ in range(max_length):
            outputs = self.model(input_ids, past_key_values=past_key_values)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # 裁剪缓存（兼容 Qwen2 的 Cache 类型）
            if past_key_values is not None and hasattr(past_key_values, "index_select"):
                seq_len = past_key_values.get_seq_length()
                if seq_len > self.window_size:
                    keep = torch.arange(seq_len - self.window_size, seq_len, device=input_ids.device)
                    past_key_values = past_key_values.index_select(keep)

            # 生成下一个 token
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        return tokenizer.decode(input_ids[0])


sparse_model = SparseCacheModel(model, window_size=512)
output = sparse_model.generate("The future of AI is")
print(output)
