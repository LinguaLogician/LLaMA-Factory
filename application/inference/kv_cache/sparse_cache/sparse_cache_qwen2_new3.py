# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: sparse_cache.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/7/30 20:37

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np


class SparseCacheModel:
    def __init__(self, model, window_size=512):
        self.model = model
        self.window_size = window_size  # 滑动窗口大小

    def generate(self, input_ids, max_length=50):
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
            # next_token = torch.argmax(logits[:, -1, :], dim=-1)
            # input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # next_token = torch.argmax(logits[:, -1, :], dim=-1)
            # input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return tokenizer.decode(input_ids[0])

if __name__ == "__main__":
    temperature = 0.95
    model_name_or_path="/home/liangtao/Models/Qwen/Qwen2-0.5B"
    messages = [
        {"role": "user", "content": "What is the future of AI"}
    ]

    messages = messages + [{"role": "assistant", "content": ""}]
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to("cuda:0")
    print(model.config)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda:0")
    eos_token_id = tokenizer.eos_token_id
    sparse_model = SparseCacheModel(model, window_size=512)

    # tokenizer.decode(input_ids[0], skip_special_tokens=True)
    input_ids = input_ids[:, :-2]
    print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
    output = sparse_model.generate(input_ids)
    print(output)
