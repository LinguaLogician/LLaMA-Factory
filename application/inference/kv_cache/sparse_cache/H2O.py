# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: H2O.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/1 14:58

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple


class H2OSparseKVCache:
    """H2O风格的稀疏K-V Cache实现"""

    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            window_size: int = 1024,
            heavy_hitter_ratio: float = 0.25,
            device: str = "cuda"
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.heavy_hitter_ratio = heavy_hitter_ratio
        self.device = device

        # 初始化缓存（实际使用中按需扩展）
        self.k_cache = None  # Shape: [batch, heads, seq_len, head_dim]
        self.v_cache = None
        self.attention_accumulator = None  # 用于累计注意力分数

    def prune(self, attention_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于累计注意力分数修剪K-V Cache，保留Heavy-Hitters"""
        batch, heads, curr_len, past_len = attention_scores.shape

        # 更新累计注意力分数（滑动窗口）
        if self.attention_accumulator is None:
            self.attention_accumulator = attention_scores.sum(dim=1)  # [batch, curr_len, past_len]
        else:
            self.attention_accumulator = torch.cat([
                self.attention_accumulator,
                attention_scores.sum(dim=1)
            ], dim=1)[:, -self.window_size:, :]  # 滑动窗口截断

        # 计算每个历史token的累计重要性（跨所有查询步）
        token_importance = self.attention_accumulator.mean(dim=1)  # [batch, past_len]

        # 选择Top-k作为Heavy-Hitters
        k = int(past_len * self.heavy_hitter_ratio)
        topk_indices = torch.topk(token_importance, k=k, dim=-1).indices  # [batch, k]

        # 创建掩码（仅保留Heavy-Hitters）
        mask = torch.zeros_like(token_importance).scatter_(
            dim=-1, index=topk_indices, value=1.0
        ).unsqueeze(1).unsqueeze(-1)  # [batch, 1, past_len, 1]

        # 应用掩码
        pruned_k = self.k_cache * mask
        pruned_v = self.v_cache * mask
        return pruned_k, pruned_v

    def update(
            self,
            new_k: torch.Tensor,
            new_v: torch.Tensor,
            attention_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新缓存并返回修剪后的K-V"""
        if self.k_cache is None:
            self.k_cache = new_k
            self.v_cache = new_v
            return new_k, new_v

        # 拼接新K/V
        self.k_cache = torch.cat([self.k_cache, new_k], dim=-2)
        self.v_cache = torch.cat([self.v_cache, new_v], dim=-2)

        # 滑动窗口截断
        if self.k_cache.shape[-2] > self.window_size:
            self.k_cache = self.k_cache[:, :, -self.window_size:, :]
            self.v_cache = self.v_cache[:, :, -self.window_size:, :]
            if self.attention_accumulator is not None:
                self.attention_accumulator = self.attention_accumulator[:, -self.window_size:, :]

        # 修剪缓存（如果提供attention_scores）
        if attention_scores is not None:
            return self.prune(attention_scores)
        return self.k_cache, self.v_cache


def load_model_and_tokenizer(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    """加载模型和分词器（需提前安装transformers和accelerate）"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


def generate_with_h2o(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        window_size: int = 1024,
        heavy_hitter_ratio: float = 0.25
) -> str:
    """使用H2O稀疏K-V Cache生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    # 初始化H2O缓存
    config = model.config
    cache = H2OSparseKVCache(
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        window_size=window_size,
        heavy_hitter_ratio=heavy_hitter_ratio,
        device=model.device
    )

    # 预填充阶段（首轮计算）
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    generated_ids = input_ids

    # 生成阶段（自回归）
    for _ in range(max_new_tokens):
        # 获取上一步的logits和注意力分数（需模型返回）
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # 获取注意力分数（需自定义模型或使用钩子）
        # 此处简化：假设能通过outputs获取最后一层的attention_probs
        attention_scores = getattr(outputs, "attentions", None)
        if attention_scores is not None:
            attention_scores = attention_scores[-1][:, :, -1:, :]  # [batch, heads, 1, past_len]

        # 更新稀疏缓存
        new_k, new_v = [], []
        for i, (k, v) in enumerate(past_key_values):
            # 假设past_key_values的每个元素是(key, value)元组
            updated_k, updated_v = cache.update(
                k[:, :, -1:, :],  # 新token的K/V
                v[:, :, -1:, :],
                attention_scores[:, i, :, :] if attention_scores is not None else None
            )
            new_k.append(updated_k)
            new_v.append(updated_v)

        # 构造新的past_key_values
        past_key_values = tuple(zip(new_k, new_v))

        # 下一步预测
        outputs = model(
            next_token,
            past_key_values=past_key_values,
            output_attentions=True  # 需要获取注意力分数
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# 示例运行
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    prompt = "The future of AI is"
    result = generate_with_h2o(
        model,
        tokenizer,
        prompt,
        max_new_tokens=50,
        window_size=512,
        heavy_hitter_ratio=0.3
    )
    print("Generated Text:", result)
