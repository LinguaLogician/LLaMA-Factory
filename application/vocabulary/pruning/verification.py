import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 用于调试CUDA错误

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置参数
model_path = "/home/liangtao/Models/Qwen/Qwen2-0.5B-VocabPruned/"
temperature = 0.7
max_new_tokens = 200


def verify_model():
    # 1. 检查词表完整性
    print("检查词表完整性...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 验证特殊token是否存在
    required_special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    for token in required_special_tokens:
        if token not in tokenizer.get_vocab():
            raise ValueError(f"缺失必要特殊token: {token}")

    # 2. 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 3. 验证设备兼容性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 4. 测试对话生成
    print("\n=== 测试对话生成 ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the future of AI?"}
    ]

    try:
        # 生成输入IDs
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # 生成attention mask
        attention_mask = torch.ones_like(input_ids)

        print("输入token数:", input_ids.shape[1])
        print("示例输入:", tokenizer.decode(input_ids[0], skip_special_tokens=False))

        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # 解码输出
        full_response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print("\n生成结果:", full_response)

        return True
    except Exception as e:
        print(f"生成失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 执行验证
    success = verify_model()

    # 额外诊断信息
    if not success:
        print("\n=== 诊断建议 ===")
        print("1. 检查词表文件是否完整")
        print("2. 确认所有特殊token在缩减后保留")
        print("3. 尝试在CPU上运行以获取更详细错误信息")
        print("4. 检查模型文件完整性 (md5sum)")

        # 检查词表大小
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        vocab_size = len(tokenizer.get_vocab())
        print(f"\n当前词表大小: {vocab_size}")

        # 检查特殊token
        print("\n特殊token映射:")
        print(tokenizer.special_tokens_map)

        # 检查模型配置
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print("\n模型配置vocab_size:", config.vocab_size)