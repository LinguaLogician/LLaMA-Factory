# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: tokenization.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/8 22:50

# from transformers import PreTrainedTokenizerFast
# import json

# 假设的 tokenizer_config.json 内容（与你提供的配置一致）
tokenizer_config = {
    "add_prefix_space": False,
    "added_tokens_decoder": {
        "151643": {"content": "<|endoftext|>", "special": True},
        "151644": {"content": "<|im_start|>", "special": True},
        "151645": {"content": "<|im_end|>", "special": True}
    },
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    "bos_token": None,
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
    "unk_token": None,
    "tokenizer_class": "Qwen2Tokenizer"
}


# 模拟一个简化的 Qwen2Tokenizer
class DummyQwen2Tokenizer:
    def __init__(self, config):
        self.config = config
        # 假设的词汇表（实际中会从文件加载）
        self.vocab = {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "<|endoftext|>": 151643,
            "user": 500,
            "\n": 101,
            "Hello": 1020,
            ",": 110,
            "how": 201, "is": 202, "the": 203, "weather": 204, "today": 205, "?": 206
        }
        self.special_tokens = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}

    def tokenize(self, text):
        tokens = []
        i = 0
        n = len(text)

        while i < n:
            # 优先检查特殊标记
            matched_special = False
            for special in self.special_tokens:
                if text.startswith(special, i):
                    tokens.append(special)
                    i += len(special)
                    matched_special = True
                    break

            if matched_special:
                continue

            # 处理普通文本（简化版：按空格和标点分割）
            if text[i].isspace():
                if text[i] == "\n":
                    tokens.append("\n")
                i += 1
            elif text[i] in {",", "?", "!"}:  # 标点符号作为独立token
                tokens.append(text[i])
                i += 1
            else:
                # 提取单词（实际分词器会用BPE/WordPiece算法）
                j = i
                while j < n and (text[j].isalnum() or text[j] in {"'", "-"}):
                    j += 1
                word = text[i:j]
                if word in self.vocab:
                    tokens.append(word)
                else:
                    # 模拟子词拆分（假设拆分为首字母+剩余部分）
                    tokens.extend([word[0], word[1:]])
                i = j

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab["<|endoftext|>"]) for token in tokens]


# 初始化模拟的分词器
tokenizer = DummyQwen2Tokenizer(tokenizer_config)

# 要分词的英文文本
text = "<|im_start|>user\nHello, how is the weather today?<|im_end|>"

# 分词过程
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# 打印结果
print("原始文本:")
print(text)
print("\n分词结果 (Tokens):")
print(tokens)
print("\n对应的 Input IDs:")
print(input_ids)
