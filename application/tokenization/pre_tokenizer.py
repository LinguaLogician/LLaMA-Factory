from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers import normalizers
from tokenizers.normalizers import NFC


def build_pre_tokenizer():
    # 1. 构建正则表达式分割预分词器
    split_pretok = pre_tokenizers.Split(
        pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        behavior="isolated",  # 注意这里是全小写
        invert=False
    )

    # 2. 构建字节级预分词器
    byte_pretok = ByteLevel(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=False
    )

    # 3. 组合成序列预分词器
    sequence_pretok = pre_tokenizers.Sequence([
        split_pretok,
        byte_pretok
    ])

    return sequence_pretok


if __name__ == "__main__":
    # 构建预分词器
    pre_tokenizer = build_pre_tokenizer()

    # 测试用例
    test_cases = [
        "I'm testing the tokenizer.",
        "Don't split this: 你好",
        "Special chars: $100, 25.5%",
        "New\nlines\r\nhandling"
    ]

    for text in test_cases:
        print(f"\n原始文本: {repr(text)}")

        # 使用HuggingFace Tokenizers的预分词器处理
        output = pre_tokenizer.pre_tokenize_str(text)

        print("预分词结果:")
        for i, (token, pos) in enumerate(output):
            print(f"  [{i}] {repr(token)} (位置: {pos})")