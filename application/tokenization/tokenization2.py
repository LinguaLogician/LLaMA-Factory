# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: tokenization2.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/9 12:47

import re
from collections import defaultdict

# 假设的 tokenizer_config.json 配置
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

# 模拟一个包含中英文子词的词汇表
vocab = {
    # 特殊标记
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|endoftext|>": 151643,

    # 英文词汇和子词
    "user": 500,
    "Hello": 1020,
    "how": 201, "is": 202, "the": 203, "weather": 204, "today": 205,
    "Hel": 1021, "lo": 1022, "wea": 2041, "ther": 2042,
    "\n": 101, ",": 110, "?": 206,

    # 中文单字和常见词
    "你": 300, "好": 301, "今天": 302, "天气": 303, "如何": 304, "助": 305, "手": 306, "是": 307,
    "，": 310, "？": 311
}

# 构建BPE合并规则（模拟子词拆分）
merge_rules = [
    ("H", "e"),  # He
    ("He", "l"),  # Hel
    ("Hel", "lo"),  # Hello
    ("w", "e"),  # we
    ("we", "a"),  # wea
    ("t", "he"),  # the
    ("th", "er"),  # ther
    ("wea", "ther"),  # weather
    ("你", "好"),  # 你好
    ("今", "天"),  # 今天
    ("天", "气"),  # 天气
    ("如", "何")  # 如何
]


class SimulatedTokenizer:
    def __init__(self, config, vocab, merge_rules):
        self.config = config
        self.vocab = vocab
        self.merge_rules = merge_rules
        self.special_tokens = set(config["additional_special_tokens"] + [config["eos_token"]])
        self.pattern = re.compile(
            r"<\|im_start\|>|<\|im_end\|>|"  # 特殊标记
            r"\w+|\S"  # 单词或单个非空白字符
        )

    def _apply_merge_rules(self, word):
        """模拟BPE子词拆分"""
        if word in self.vocab:
            return [word]

        # 尝试应用合并规则
        for a, b in self.merge_rules:
            if a + b in word:
                parts = word.split(a + b)
                result = []
                for i, part in enumerate(parts):
                    if part:
                        result.extend(self._apply_merge_rules(part))
                    if i < len(parts) - 1:
                        result.append(a + b)
                return result

        # 最终拆分为单个字符
        return list(word)

    def tokenize(self, text):
        tokens = []
        for token in self.pattern.finditer(text):
            token = token.group()

            # 处理特殊标记
            if token in self.special_tokens:
                tokens.append(token)
                continue

            # 处理英文和中文
            if re.match(r'^[\w\-\']+$', token):  # 英文单词
                subtokens = self._apply_merge_rules(token)
                tokens.extend(subtokens)
            else:  # 中文或其他字符
                for char in token:
                    if char in self.vocab:
                        tokens.append(char)
                    else:
                        tokens.append("<|endoftext|>")  # 未知字符用EOS代替

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab["<|endoftext|>"]) for token in tokens]


# 初始化分词器
tokenizer = SimulatedTokenizer(tokenizer_config, vocab, merge_rules)

# 测试混合中英文文本
texts = [
    "<|im_start|>user\nHeHellHe, how is the weather today?<|im_end|>",  # 英文
    "<|im_start|>system\n你好，今天天气如何？<|im_end|>",  # 中文
    "<|im_start|>user\nHello你好<|im_end|>"  # 混合
]

for text in texts:
    print(f"\n原始文本: {text}")
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("分词结果:")
    print(tokens)
    print("Input IDs:")
    print(input_ids)
