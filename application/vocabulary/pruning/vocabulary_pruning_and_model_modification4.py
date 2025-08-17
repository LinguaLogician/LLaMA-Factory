import json
import os
import shutil
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file as safe_save
from safetensors import safe_open
from collections import OrderedDict


def is_ascii(s):
    """检查字符串是否只包含ASCII字符"""
    return all(ord(c) < 128 for c in s)


def prune_vocab_and_remap(vocab):
    """
    缩减词表并重新映射ID
    返回: (pruned_vocab, old_to_new_id_map, special_tokens)
    """
    # 先筛选出ASCII token并按原始ID排序
    ascii_items = sorted([(k, v) for k, v in vocab.items() if is_ascii(k)], key=lambda x: x[1])

    # 创建新词表和ID映射
    pruned_vocab = OrderedDict()
    old_to_new = {}
    special_tokens = []

    # 首先处理特殊token（确保ID不变）
    for token, old_id in ascii_items:
        if token.startswith("<|") and token.endswith("|>"):  # 识别特殊token
            pruned_vocab[token] = old_id
            old_to_new[old_id] = old_id
            special_tokens.append(token)

    # 然后处理普通token（重新分配ID）
    next_id = max(old_to_new.values()) + 1 if old_to_new else 0
    for token, old_id in ascii_items:
        if token not in pruned_vocab:
            pruned_vocab[token] = next_id
            old_to_new[old_id] = next_id
            next_id += 1

    return pruned_vocab, old_to_new, special_tokens


def prune_merges(merges):
    """缩减merges列表，保留只包含ASCII字符的merge"""
    return [merge for merge in merges if all(is_ascii(part) for part in merge.split())]


def update_tokenizer_config(tokenizer_config_path, special_tokens):
    """更新tokenizer_config.json确保特殊token一致"""
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 更新special_tokens_map
    if "special_tokens_map" in config:
        config["special_tokens_map"] = {
            k: v for k, v in config["special_tokens_map"].items()
            if v in special_tokens
        }

    return config


def adjust_model_embeddings(model, old_to_new_id_map, new_vocab_size, original_dtype):
    """
    调整模型嵌入层并保持原始精度
    """
    # 获取原始嵌入层
    old_embeddings = model.get_input_embeddings()
    old_vocab_size, embedding_dim = old_embeddings.weight.shape

    # 创建新的嵌入层（保持原始精度）
    new_embeddings = torch.nn.Embedding(new_vocab_size, embedding_dim, dtype=original_dtype)

    # 复制保留的token的embedding
    with torch.no_grad():
        for old_id, new_id in old_to_new_id_map.items():
            new_embeddings.weight[new_id] = old_embeddings.weight[old_id]

        # 初始化未映射的位置（使用保留token的平均值）
        kept_embeddings = torch.stack([
            old_embeddings.weight[old_id]
            for old_id in old_to_new_id_map.keys()
        ])
        mean_embedding = kept_embeddings.mean(dim=0)
        for i in range(new_vocab_size):
            if i not in old_to_new_id_map.values():
                new_embeddings.weight[i] = mean_embedding

    # 更新模型输入嵌入
    model.set_input_embeddings(new_embeddings)

    # 处理输出层（无论是否绑定）
    if model.get_output_embeddings() is not None:
        old_output = model.get_output_embeddings()
        has_bias = hasattr(old_output, "bias") and old_output.bias is not None

        # 创建新的输出层
        new_output = torch.nn.Linear(
            embedding_dim,
            new_vocab_size,
            bias=has_bias,
            dtype=original_dtype
        )

        # 复制权重
        with torch.no_grad():
            if model.config.tie_word_embeddings:
                # 如果绑定，直接使用新的输入嵌入权重
                new_output.weight = torch.nn.Parameter(new_embeddings.weight.clone())
            else:
                # 如果不绑定，复制保留的权重
                for old_id, new_id in old_to_new_id_map.items():
                    new_output.weight[new_id] = old_output.weight[old_id]

                # 初始化未映射的位置
                kept_output_weights = torch.stack([
                    old_output.weight[old_id]
                    for old_id in old_to_new_id_map.keys()
                ])
                mean_output = kept_output_weights.mean(dim=0)
                for i in range(new_vocab_size):
                    if i not in old_to_new_id_map.values():
                        new_output.weight[i] = mean_output

            # 处理偏置（如果有）
            if has_bias:
                new_output.bias.data.copy_(old_output.bias.data[:new_vocab_size])

        model.set_output_embeddings(new_output)

    # 更新config
    model.config.vocab_size = new_vocab_size

    return model


def save_model_safely(model, save_path, original_metadata=None):
    """
    安全保存模型，处理共享权重问题
    """
    state_dict = model.state_dict()

    # 处理共享权重问题
    if model.config.tie_word_embeddings:
        # 解除权重绑定
        lm_head_weight = state_dict["lm_head.weight"].clone()
        state_dict["lm_head.weight"] = lm_head_weight

    # 设置metadata
    metadata = {
        "format": "pt",
        "vocab_size": str(model.config.vocab_size),
    }

    # 添加原始metadata（如果存在）
    if original_metadata and isinstance(original_metadata, dict):
        metadata.update({
            k: v for k, v in original_metadata.items()
            if k in {"description", "author", "date"}
        })

    # 保存为safetensors
    safe_save(state_dict, os.path.join(save_path, "model.safetensors"), metadata)


def get_model_metadata(model_path):
    """获取原始模型的metadata"""
    model_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(model_file):
        with safe_open(model_file, framework="pt") as f:
            return f.metadata
    return None


def main():
    # 原始模型路径
    model_path = "/home/liangtao/Models/Qwen/Qwen2-0.5B/"
    # 新模型路径
    new_model_path = "/home/liangtao/Models/Qwen/Qwen2-0.5B-VocabPruned/"

    # 创建新目录
    os.makedirs(new_model_path, exist_ok=True)

    # 1. 获取原始metadata
    original_metadata = get_model_metadata(model_path)

    # 2. 加载原始模型和tokenizer（自动检测精度）
    print("加载原始模型...")
    original_dtype = torch.bfloat16  # 默认假设为float16
    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
            if config.get("torch_dtype") == "float32":
                original_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=original_dtype,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 3. 处理tokenizer.json
    print("处理tokenizer...")
    tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # 缩减vocab并建立ID映射
    pruned_vocab, old_to_new_id_map, special_tokens = prune_vocab_and_remap(tokenizer_data['model']['vocab'])
    pruned_merges = prune_merges(tokenizer_data['model']['merges'])
    new_vocab_size = len(pruned_vocab)

    # 更新tokenizer.json
    tokenizer_data['model']['vocab'] = pruned_vocab
    tokenizer_data['model']['merges'] = pruned_merges
    tokenizer_data['added_tokens'] = [
        tok for tok in tokenizer_data.get('added_tokens', [])
        if tok['content'] in special_tokens
    ]

    # 保存新的tokenizer文件
    with open(os.path.join(new_model_path, "tokenizer.json"), 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    # 4. 处理其他tokenizer相关文件
    print("处理配套文件...")
    # vocab.json (如果有)
    if os.path.exists(os.path.join(model_path, "vocab.json")):
        with open(os.path.join(model_path, "vocab.json"), 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        pruned_vocab_json, _, _ = prune_vocab_and_remap(vocab_data)
        with open(os.path.join(new_model_path, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(pruned_vocab_json, f, ensure_ascii=False, indent=2)

    # merges.txt (如果有)
    if os.path.exists(os.path.join(model_path, "merges.txt")):
        with open(os.path.join(model_path, "merges.txt"), 'r', encoding='utf-8') as f:
            merges_lines = f.readlines()
        pruned_merges_lines = [line for line in merges_lines if all(is_ascii(part) for part in line.strip().split())]
        with open(os.path.join(new_model_path, "merges.txt"), 'w', encoding='utf-8') as f:
            f.writelines(pruned_merges_lines)

    # tokenizer_config.json
    if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        new_tokenizer_config = update_tokenizer_config(
            os.path.join(model_path, "tokenizer_config.json"),
            special_tokens
        )
        with open(os.path.join(new_model_path, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(new_tokenizer_config, f, ensure_ascii=False, indent=2)

    # 5. 调整模型
    print("调整模型结构...")
    model = adjust_model_embeddings(model, old_to_new_id_map, new_vocab_size, original_dtype)

    # 6. 保存配置文件
    print("保存配置文件...")
    # config.json
    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['vocab_size'] = new_vocab_size
        with open(os.path.join(new_model_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    # generation_config.json
    if os.path.exists(os.path.join(model_path, "generation_config.json")):
        shutil.copy2(
            os.path.join(model_path, "generation_config.json"),
            os.path.join(new_model_path, "generation_config.json")
        )

    # 7. 安全保存模型权重
    print("保存模型权重...")
    save_model_safely(model, new_model_path, original_metadata)

    # 8. 验证
    print("验证结果...")
    print(f"\n{'=' * 40}\n词表缩减结果:")
    print(f"原始词表大小: {len(tokenizer_data['model']['vocab']):,}")
    print(f"新词表大小: {new_vocab_size:,}")
    print(f"移除的非ASCII token数量: {len(tokenizer_data['model']['vocab']) - new_vocab_size:,}")

    # 检查文件大小
    original_size = os.path.getsize(os.path.join(model_path, "model.safetensors")) / 1024 ** 2
    new_size = os.path.getsize(os.path.join(new_model_path, "model.safetensors")) / 1024 ** 2
    print(f"\n文件大小变化:")
    print(f"原始: {original_size:.1f}MB")
    print(f"新: {new_size:.1f}MB")
    print(f"变化: {'+' if new_size > original_size else ''}{new_size - original_size:.1f}MB")

    print(f"\n处理完成！新模型已保存到: {new_model_path}")


if __name__ == "__main__":
    main()