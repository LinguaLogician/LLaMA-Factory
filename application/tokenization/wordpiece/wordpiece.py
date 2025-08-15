# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: wordpiece.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/14 14:48

"""
WordPiece Tokenization Algorithm - Complete Implementation
"""
import collections
import re
from typing import Dict, List, Tuple


class WordPieceTokenizer:
    def __init__(self, corpus: List[str], vocab_size: int = 30):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = {}
        self.special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[MASK]"]

        self._train()

    def _preprocess(self, text: str) -> str:
        """Text normalization"""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)  # Keep only alphanumeric
        return text

    def _initialize_vocab(self) -> Dict[str, int]:
        """Initialize vocabulary with characters and word boundaries"""
        vocab = collections.defaultdict(int)
        for text in self.corpus:
            text = self._preprocess(text)
            for word in text.split():
                # Add word boundary symbol
                token = " ".join(list(word)) + " </w>"
                vocab[token] += 1
        return vocab

    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get frequency of adjacent symbol pairs"""
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _get_merge_cost(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> int:
        """Calculate how many tokens would be reduced by merging this pair"""
        cost = 0
        first, second = pair
        pattern = re.compile(r"(?<!\S)" + re.escape(first + " " + second) + r"(?!\S)")

        for word, freq in vocab.items():
            matches = pattern.findall(word)
            cost += len(matches) * freq  # Each merge reduces 1 token

        return cost

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge the highest frequency pair in vocabulary"""
        new_vocab = {}
        first, second = pair
        pattern = re.compile(r"(?<!\S)" + re.escape(first + " " + second) + r"(?!\S)")

        for word in vocab:
            new_word = pattern.sub(first + second, word)
            new_vocab[new_word] = vocab[word]

        return new_vocab

    def _train(self) -> None:
        """Train WordPiece vocabulary"""
        vocab = self._initialize_vocab()

        # Add single characters to final vocab
        self.vocab = list(set('abcdefghijklmnopqrstuvwxyz0123456789'))
        self.vocab.extend(self.special_tokens)

        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            # Find the best merge pair
            best_pair = None
            max_cost = -1
            for pair in pairs:
                cost = self._get_merge_cost(pair, vocab)
                if cost > max_cost:
                    max_cost = cost
                    best_pair = pair

            if not best_pair:
                break

            # Perform merge
            merged = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged
            self.vocab.append(merged)
            vocab = self._merge_vocab(best_pair, vocab)

        print(f"Training completed. Vocabulary size: {len(self.vocab)}")
        print(f"Merges performed: {len(self.merges)}")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize new text using trained vocabulary"""
        text = self._preprocess(text)
        words = text.split()
        tokens = []

        for word in words:
            word = word + "</w>"
            start = 0
            sub_tokens = []

            while start < len(word):
                end = len(word)
                found = False

                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr  # Mark as non-starting substring

                    if substr in self.vocab:
                        sub_tokens.append(substr)
                        found = True
                        start = end
                        break
                    else:
                        end -= 1

                if not found:
                    sub_tokens.append("[UNK]")
                    break

            tokens.extend(sub_tokens)

        return tokens


# Example Usage
if __name__ == "__main__":
    # Sample corpus
    corpus = [
        "Deep learning is a subset of machine learning.",
        "WordPiece is used in BERT tokenization.",
        "Subword tokenization handles rare words better."
    ]

    # Initialize and train tokenizer
    tokenizer = WordPieceTokenizer(corpus, vocab_size=30)

    # Test tokenization
    test_text = "WordPiece handles rare words"
    tokens = tokenizer.tokenize(test_text)
    print(f"\nTokenization example:")
    print(f"Input: '{test_text}'")
    print(f"Tokens: {tokens}")

    # Show vocabulary
    print("\nVocabulary samples:")
    print(tokenizer.vocab[:10], "...")
