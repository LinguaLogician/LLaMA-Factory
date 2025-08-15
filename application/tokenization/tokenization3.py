# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: tokenization2.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/9 12:47

import re
from collections import defaultdict
tokenizer_config = {
    "add_prefix_space": False,
    "added_tokens_decoder": {
        "151643": {"content": "<|endoftext|>", "special": True},
        "151644": {"content": "<|im_start|>", "special": True},
        "151645": {"content": "<|im_end|>", "special": True}
    },
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
    "model_max_length": 32768,
    "tokenizer_class": "Qwen2Tokenizer"
}
# 初始词汇表（所有基础字符）
base_vocab = {
    # 特殊标记
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|endoftext|>": 151643,

    # 基础字符（英文和中文）
    **{chr(i): i for i in range(32, 127)},  # ASCII字符
    **{chr(i): i + 10000 for i in range(19968, 40959)},  # 常用汉字
    "\n": 101, ",": 110, "?": 206, "，": 310, "？": 311
}

# BPE合并规则（记录符号对合并操作）
merge_rules = [
    ("H", "e"),  # 合并为 He
    ("He", "l"),  # 合并为 Hel
    ("Hel", "lo"),  # 合并为 Hello
    ("t", "h"),  # 合并为 th
    ("th", "e"),  # 合并为 the
    ("w", "e"),  # 合并为 we
    ("we", "a"),  # 合并为 wea
    ("e", "a"),  # 合并为 ea
    ("a", "t"),  # 合并为 at
    ("th", "er"),  # 合并为 ther
    ("你", "好"),  # 合并为 你好
    ("今", "天"),  # 合并为 今天
]


class BPETokenizer:
    def __init__(self, config, base_vocab, merge_rules):
        self.config = config
        self.vocab = base_vocab.copy()
        self.merge_rules = merge_rules
        self.special_tokens = set(config["additional_special_tokens"] + [config["eos_token"]])
        self.pattern = re.compile(
            r"<\|im_start\|>|<\|im_end\|>|"  # 特殊标记
            r"\w+|\S"  # 单词或单个非空白字符
        )

        # 根据合并规则动态扩展词汇表
        for a, b in merge_rules:
            merged = a + b
            if merged not in self.vocab:
                self.vocab[merged] = max(self.vocab.values()) + 1

    def _apply_bpe(self, word):
        """应用BPE规则拆分单词"""
        if word in self.vocab:
            return [word]

        # 初始化为单个字符
        subwords = list(word)

        # 按优先级应用所有合并规则
        for a, b in self.merge_rules:
            merged = a + b
            new_subwords = []
            i = 0
            while i < len(subwords):
                if i < len(subwords) - 1 and subwords[i] == a and subwords[i + 1] == b:
                    new_subwords.append(merged)
                    i += 2
                else:
                    new_subwords.append(subwords[i])
                    i += 1
            subwords = new_subwords

        return subwords

    def tokenize(self, text):
        tokens = []
        for token in self.pattern.finditer(text):
            token = token.group()

            if token in self.special_tokens:
                tokens.append(token)
                continue

            if re.match(r'^[\w\-\']+$', token):  # 英文单词
                tokens.extend(self._apply_bpe(token))
            else:  # 中文或其他字符
                for char in token:
                    if char in self.vocab:
                        tokens.append(char)
                    else:
                        tokens.append("<|endoftext|>")
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab["<|endoftext|>"]) for token in tokens]


# 初始化分词器
tokenizer = BPETokenizer(tokenizer_config, base_vocab, merge_rules)

# 测试
text = "<|im_start|>user\nHello, the weather is nice today?<|im_end|>"
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)

print("分词结果:", tokens)
print("Input IDs:", input_ids)
