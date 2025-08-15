# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: sentencepiece.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/14 15:20

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentencePiece BPE 算法纯 Python 实现

实现了以下核心功能:
1. BPE 算法训练
2. 文本编码(子词分割)
3. 文本解码
4. 词汇表管理
"""

import re
import collections
import heapq
from typing import List, Dict, Tuple, Set, Optional


class BPETokenizer:
    def __init__(self):
        self.vocab: Dict[str, int] = {}  # 子词到ID的映射
        self.inv_vocab: Dict[int, str] = {}  # ID到子词的映射
        self.merges: Dict[Tuple[str, str], str] = {}  # 合并操作记录
        self.special_tokens: Dict[str, int] = {}  # 特殊标记
        self.max_token_length: int = 0  # 最长子词长度(用于优化编码)

        # 初始化特殊标记
        self.add_special_token("<pad>", 0)
        self.add_special_token("<unk>", 1)
        self.add_special_token("<bos>", 2)
        self.add_special_token("<eos>", 3)

    def add_special_token(self, token: str, token_id: Optional[int] = None) -> None:
        """添加特殊标记"""
        if token_id is None:
            token_id = max(self.special_tokens.values()) + 1 if self.special_tokens else 0
        self.special_tokens[token] = token_id
        self.vocab[token] = token_id
        self.inv_vocab[token_id] = token

    def train(self,
              text: str,
              vocab_size: int = 3000,
              min_frequency: int = 2,
              lowercase: bool = False,
              verbose: bool = False) -> None:
        """
        训练 BPE 模型

        参数:
            text: 训练文本
            vocab_size: 目标词汇表大小
            min_frequency: 最小出现频率
            lowercase: 是否转换为小写
            verbose: 是否打印训练过程
        """
        if lowercase:
            text = text.lower()

        # 1. 预处理文本并统计初始词汇(字符级别)
        words, word_counts = self._preprocess_text(text)
        vocab = self._initialize_vocab(words, word_counts)

        # 2. 迭代合并最高频的字符对
        while len(vocab) + len(self.special_tokens) < vocab_size:
            # 统计字符对频率
            pairs = self._get_stats(words, word_counts)
            if not pairs:
                break

            # 找到最高频的字符对
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]

            if best_freq < min_frequency:
                if verbose:
                    print(f"停止训练，没有足够高频的字符对 (频率 < {min_frequency})")
                break

            # 合并字符对
            if verbose:
                print(f"合并: {best_pair} (频率: {best_freq})")

            vocab, words = self._merge_vocab(best_pair, vocab, words, word_counts)
            self.merges[best_pair] = best_pair[0] + best_pair[1]

            # 更新最长token长度
            new_token = best_pair[0] + best_pair[1]
            if len(new_token) > self.max_token_length:
                self.max_token_length = len(new_token)

        # 3. 构建最终词汇表
        self._build_final_vocab(vocab)

        if verbose:
            print(f"训练完成，词汇表大小: {len(self.vocab)}")
            print(f"最长子词长度: {self.max_token_length}")

    def _preprocess_text(self, text: str) -> Tuple[List[List[str]], Dict[str, int]]:
        """预处理文本，分割为单词并统计频率"""
        # 简单分词(按空格分割)
        words = re.findall(r"\S+|\n", text)

        # 统计词频
        word_counts = collections.Counter(words)

        # 将单词分割为字符，添加结束符 </w>
        processed_words = []
        for word in word_counts.keys():
            chars = list(word) + ["</w>"]
            processed_words.append(chars)

        return processed_words, word_counts

    def _initialize_vocab(self,
                          words: List[List[str]],
                          word_counts: Dict[str, int]) -> Dict[str, int]:
        """初始化词汇表(字符级别)"""
        vocab = collections.defaultdict(int)

        # 统计字符频率
        for word, count in zip(words, word_counts.values()):
            for char in word:
                vocab[char] += count

        return vocab

    def _get_stats(self,
                   words: List[List[str]],
                   word_counts: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """统计相邻字符对的频率"""
        pairs = collections.defaultdict(int)

        for word, count in zip(words, word_counts.values()):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += count

        return pairs

    def _merge_vocab(self,
                     pair: Tuple[str, str],
                     vocab: Dict[str, int],
                     words: List[List[str]],
                     word_counts: Dict[str, int]) -> Tuple[Dict[str, int], List[List[str]]]:
        """合并指定的字符对"""
        new_token = pair[0] + pair[1]
        vocab[new_token] = 0

        # 更新词汇表频率
        for word, count in zip(words, word_counts.values()):
            i = 0
            while i < len(word) - 1:
                if word[i] == pair[0] and word[i + 1] == pair[1]:
                    # 合并字符对
                    word[i] = new_token
                    word.pop(i + 1)
                    vocab[new_token] += count
                else:
                    i += 1

        return vocab, words

    def _build_final_vocab(self, vocab: Dict[str, int]) -> None:
        """构建最终词汇表"""
        # 分配ID，确保特殊标记有固定ID
        next_id = max(self.special_tokens.values()) + 1 if self.special_tokens else 0

        # 按频率排序
        sorted_vocab = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))

        # 构建词汇表
        for token, freq in sorted_vocab:
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.inv_vocab[next_id] = token
                next_id += 1

    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID序列"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.special_tokens["<unk>"]) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """将token ID序列解码为文本"""
        tokens = [self.inv_vocab.get(id, "<unk>") for id in ids]
        return self.detokenize(tokens)

    def tokenize(self, text: str) -> List[str]:
        """将文本分割为子词token"""
        # 预处理: 按空格分割单词
        words = re.findall(r"\S+|\n", text)

        tokens = []
        for word in words:
            # 添加结束符
            word_chars = list(word) + ["</w>"]

            # 初始化为单个字符
            sub_tokens = word_chars

            # 迭代应用合并操作
            changed = True
            while changed and len(sub_tokens) > 1:
                changed = False

                # 寻找最长的可合并pair
                best_pair = None
                best_pos = -1
                best_length = 0

                for i in range(len(sub_tokens) - 1):
                    pair = (sub_tokens[i], sub_tokens[i + 1])
                    merged = self.merges.get(pair, None)

                    if merged is not None and len(merged) > best_length:
                        best_pair = pair
                        best_pos = i
                        best_length = len(merged)

                if best_pair is not None:
                    # 执行合并
                    merged_token = self.merges[best_pair]
                    sub_tokens[best_pos] = merged_token
                    del sub_tokens[best_pos + 1]
                    changed = True

            # 处理未知token
            for token in sub_tokens:
                if token not in self.vocab:
                    # 尝试分解为更小的已知token
                    decomposed = self._decompose_unknown(token)
                    tokens.extend(decomposed)
                else:
                    tokens.append(token)

        return tokens

    def _decompose_unknown(self, token: str) -> List[str]:
        """尝试分解未知token为已知子词"""
        # 如果以</w>结尾，先处理前面的部分
        end_with_w = token.endswith("</w>")
        if end_with_w:
            core_token = token[:-4]
        else:
            core_token = token

        # 尝试按最长匹配分解
        decomposed = []
        start = 0
        n = len(core_token)

        while start < n:
            end = min(n, start + self.max_token_length)
            found = False

            # 从最长可能开始找
            while end > start:
                substr = core_token[start:end]
                if substr in self.vocab:
                    decomposed.append(substr)
                    start = end
                    found = True
                    break
                end -= 1

            if not found:
                # 无法分解，使用<unk>
                decomposed.append("<unk>")
                start += 1

        if end_with_w:
            decomposed.append("</w>")

        return decomposed

    def detokenize(self, tokens: List[str]) -> str:
        """将token序列合并为文本"""
        text = ""
        for token in tokens:
            if token == "</w>":
                text += " "
            else:
                text += token
        return text.strip()

    def save(self, file_prefix: str) -> None:
        """保存模型到文件"""
        import json

        # 保存词汇表
        with open(f"{file_prefix}.vocab", "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()},
                "special_tokens": self.special_tokens,
                "max_token_length": self.max_token_length
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_prefix: str) -> "BPETokenizer":
        """从文件加载模型"""
        import json

        tokenizer = cls()

        with open(f"{file_prefix}.vocab", "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = {v: k for k, v in data["vocab"].items()}
        tokenizer.merges = {
            tuple(k.split(" ")): v for k, v in data["merges"].items()
        }
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.max_token_length = data["max_token_length"]

        return tokenizer


def example_usage():
    """使用示例"""
    # 示例文本
    text = """SentencePiece是一个无监督的文本tokenizer和detokenizer。
    它主要用于基于神经网络的文本生成系统。
    实现了BPE和unigram语言模型。
    可以不需要预处理(如tokenization)直接从原始句子训练。"""

    # 1. 训练tokenizer
    print("=== 训练 BPE Tokenizer ===")
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=500, verbose=True)

    # 2. 测试编码/解码
    print("\n=== 测试编码/解码 ===")
    test_text = "SentencePiece可以直接从原始文本训练tokenizer。"

    print(f"原始文本: {test_text}")

    # 编码为token
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokenized: {tokens}")

    # 编码为ID
    ids = tokenizer.encode(test_text)
    print(f"Encoded IDs: {ids}")

    # 解码
    decoded = tokenizer.decode(ids)
    print(f"Decoded text: {decoded}")

    # 3. 保存和加载模型
    print("\n=== 测试模型保存/加载 ===")
    tokenizer.save("example_bpe")

    loaded_tokenizer = BPETokenizer.load("example_bpe")
    reloaded_ids = loaded_tokenizer.encode(test_text)
    print(f"Reloaded encoded IDs: {reloaded_ids}")
    print(f"IDs匹配: {ids == reloaded_ids}")


if __name__ == '__main__':
    example_usage()
