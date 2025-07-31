# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: sparse_cache.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/7/30 20:37

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


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

            # 稀疏化：仅保留最近window_size个token的K-V
            new_past = []
            for layer_k, layer_v in past_key_values:
                truncated_k = layer_k[:, :, -self.window_size:, :]
                truncated_v = layer_v[:, :, -self.window_size:, :]
                new_past.append((truncated_k, truncated_v))
            past_key_values = tuple(new_past)

            # 生成下一个token
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        return tokenizer.decode(input_ids[0])


sparse_model = SparseCacheModel(model, window_size=512)
output = sparse_model.generate("The future of AI is")
print(output)
