import os
import json
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file


def is_valid_token(token):
    """检查token是否只包含ASCII字符或Ġ(空格)"""
    if token.startswith("<|") and token.endswith("|>"):  # 保留特殊token
        return True
    try:
        for char in token:
            if char == 'Ġ':  # 允许空格
                continue
            if char == 'Ċ':  # 允许空格
                continue
            if ord(char) > 127:  # 非ASCII
                return False
        return True
    except:
        return False


def prune_vocab_and_merges(tokenizer_path):
    """缩减词表和merges"""
    with open(os.path.join(tokenizer_path, "tokenizer.json"), "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    original_vocab = tokenizer_json["model"]["vocab"]
    kept_tokens = [token for token in original_vocab if is_valid_token(token)]

    special_tokens = [token["content"] for token in tokenizer_json["added_tokens"]]
    normal_tokens = [token for token in kept_tokens if token not in special_tokens]
    normal_tokens_sorted = sorted(normal_tokens, key=lambda x: original_vocab[x])

    normal_vocab = {token: idx for idx, token in enumerate(normal_tokens_sorted)}
    new_vocab = {token: idx for idx, token in enumerate(normal_tokens_sorted + special_tokens)}
    original_merges = tokenizer_json["model"]["merges"]
    pruned_merges = [merge for merge in original_merges if all(is_valid_token(part) for part in merge.split())]

    tokenizer_json["model"]["vocab"] = normal_vocab
    tokenizer_json["model"]["merges"] = pruned_merges

    for token_info in tokenizer_json["added_tokens"]:
        token = token_info["content"]
        if token in new_vocab:
            token_info["id"] = new_vocab[token]
    new_added_tokens = []
    for token in special_tokens:
        new_added_tokens.append({
            "id": new_vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        })
    tokenizer_json["added_tokens"] = new_added_tokens

    print(f"原始词表大小: {len(original_vocab)}")
    print(f"缩减后词表大小: {len(new_vocab)}")
    print(f"原始merges数量: {len(original_merges)}")
    print(f"缩减后merges数量: {len(pruned_merges)}")

    old_to_new = {}
    for new_id, token in new_vocab.items():
        if token in original_vocab:
            old_to_new[original_vocab[token]] = new_id
    return tokenizer_json, new_vocab, special_tokens, old_to_new


def update_tokenizer_config(tokenizer_path, new_vocab, special_tokens):
    """更新tokenizer_config.json"""
    with open(os.path.join(tokenizer_path, "tokenizer_config.json"), "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    added_tokens_decoder = {}
    for token in special_tokens:
        added_tokens_decoder[new_vocab[token]]={
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        }
    tokenizer_config["added_tokens_decoder"] = added_tokens_decoder
    return tokenizer_config


def update_config_files(tokenizer_path, new_vocab, tokenizer_config):
    """更新config.json和generation_config.json"""
    with open(os.path.join(tokenizer_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    config["vocab_size"] = len(new_vocab)

    for token_name in ["eos_token", "bos_token"]:
        if f"{token_name}_id" in config:
            token = tokenizer_config.get("eos_token", None)
            if token and token in new_vocab:
                config[f"{token_name}_id"] = new_vocab[token]

    with open(os.path.join(tokenizer_path, "generation_config.json"), "r", encoding="utf-8") as f:
        generation_config = json.load(f)

    for token_name in ["eos_token_id", "bos_token_id"]:
        if token_name in config:
            generation_config[token_name] = config[token_name]

    return config, generation_config


def adjust_model_embeddings(model_path, old_to_new_id_map, new_vocab_size, original_dtype):
    """
    调整模型嵌入层并保持原始精度
    """
    # 获取原始嵌入层
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=original_dtype)
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


def resize_model_embeddings(model_path, new_vocab_size, original_dtype=torch.bfloat16):
    """调整模型的embedding层大小并保持bfloat16精度"""
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=original_dtype)
    original_vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size

    print(f"原始模型词表大小: {original_vocab_size}")
    print(f"新词表大小: {new_vocab_size}")
    print(f"Embedding 层精度: {original_dtype}")

    if original_vocab_size == new_vocab_size:
        print("词表大小未改变，无需调整embedding层")
        return model

    old_embeddings = model.get_input_embeddings()
    old_weight = old_embeddings.weight.data

    new_embeddings = torch.nn.Embedding(new_vocab_size, hidden_size, dtype=original_dtype)
    new_embeddings = new_embeddings.to(old_weight.device)

    min_size = min(original_vocab_size, new_vocab_size)
    new_embeddings.weight.data[:min_size] = old_weight[:min_size]

    if new_vocab_size > original_vocab_size:
        new_embeddings.weight.data[original_vocab_size:] = torch.normal(
            mean=0.0,
            std=model.config.initializer_range,
            size=(new_vocab_size - original_vocab_size, hidden_size),
            device=old_weight.device,
            dtype=original_dtype
        )

    model.set_input_embeddings(new_embeddings)

    if model.config.tie_word_embeddings:
        # 创建共享权重的输出层
        model.lm_head = torch.nn.Linear(
            hidden_size,
            new_vocab_size,
            bias=False,
            dtype=original_dtype
        ).to(old_weight.device)
        model.lm_head.weight = model.get_input_embeddings().weight
        print("输入/输出 Embedding 权重已共享")
    else:
        old_output_embeddings = model.get_output_embeddings()
        if old_output_embeddings is not None:
            old_output_weight = old_output_embeddings.weight.data
            new_output_embeddings = torch.nn.Linear(
                hidden_size,
                new_vocab_size,
                bias=False,
                dtype=original_dtype
            ).to(old_output_weight.device)
            new_output_embeddings.weight.data[:min_size] = old_output_weight[:min_size]

            if new_vocab_size > original_vocab_size:
                new_output_embeddings.weight.data[original_vocab_size:] = torch.normal(
                    mean=0.0,
                    std=model.config.initializer_range,
                    size=(new_vocab_size - original_vocab_size, hidden_size),
                    device=old_output_weight.device,
                    dtype=original_dtype
                )
            model.set_output_embeddings(new_output_embeddings)
            print("输入/输出 Embedding 权重未共享，已分别调整")

    model.config.vocab_size = new_vocab_size

    return model


def save_model_safely(model, output_path):
    """安全保存模型，处理共享权重问题"""
    # 创建临时状态字典，解除权重共享
    state_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in model.state_dict().items()}

    # 保存模型配置
    model.save_pretrained(output_path, state_dict=state_dict, safe_serialization=True)

    print("模型已安全保存")


def main():
    model_path = "/home/liangtao/Models/Qwen/Qwen2-0.5B"
    output_path = "/home/liangtao/Models/Qwen/Qwen2-0.5B-VocabPruned"
    original_dtype = torch.bfloat16

    os.makedirs(output_path, exist_ok=True)

    print("=" * 50)
    print("开始缩减词表和merges...")
    tokenizer_json, new_vocab, special_tokens, old_to_new = prune_vocab_and_merges(model_path)

    print("=" * 50)
    print("更新tokenizer_config.json...")
    tokenizer_config = update_tokenizer_config(model_path, new_vocab, special_tokens)

    print("=" * 50)
    print("更新config.json和generation_config.json...")
    config, generation_config = update_config_files(model_path, new_vocab, tokenizer_config)

    print("=" * 50)
    print("调整模型embedding层...")
    model = adjust_model_embeddings(model_path, old_to_new, len(new_vocab), original_dtype)
    print("=" * 50)
    print("保存文件到新目录...")

    with open(os.path.join(output_path, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_path, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(generation_config, f, ensure_ascii=False, indent=2)

    for file in ["merges.txt", "vocab.json"]:
        if os.path.exists(os.path.join(model_path, file)):
            shutil.copy2(os.path.join(model_path, file), os.path.join(output_path, file))

    print("=" * 50)
    print("保存模型...")
    save_model_safely(model, output_path)

    print("=" * 50)
    print("所有操作完成！新模型已保存到:", output_path)
    print(f"词表大小: {len(new_vocab)}")
    print(f"模型精度: {original_dtype}")


if __name__ == "__main__":
    main()