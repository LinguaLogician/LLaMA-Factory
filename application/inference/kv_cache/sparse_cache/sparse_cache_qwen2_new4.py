import os

from sympy.physics.units import temperature

from llamafactory.hparams import get_infer_args

# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: sparse_cache.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/7/30 20:37

from transformers import AutoTokenizer
import torch

from llamafactory.chat import ChatModel
from llamafactory.chat.hf_engine import HuggingfaceEngine
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SparseCacheModel:
    def __init__(self, model, window_size=512):
        self.model = model
        self.window_size = window_size  # 滑动窗口大小

    def generate(self, gen_kwargs, max_length=50):
        input_ids = gen_kwargs["inputs"]
        past_key_values = None
        eos_token_id = tokenizer.eos_token_id
        # generate_output = self.model.generate(**gen_kwargs)
        #
        # print("=============\n",tokenizer.decode(generate_output['sequences'][0]),"=============\n")

        for _ in range(max_length):
            outputs = self.model(input_ids, past_key_values=past_key_values)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            if past_key_values is not None and hasattr(past_key_values, "index_select"):
                seq_len = past_key_values.get_seq_length()
                if seq_len > self.window_size:
                    keep = torch.arange(seq_len - self.window_size, seq_len, device=input_ids.device)
                    past_key_values = past_key_values.index_select(keep)

            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # next_token = torch.argmax(logits[:, -1, :], dim=-1)
            # input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return tokenizer.decode(input_ids[0])

if __name__ == "__main__":
    model_name_or_path = "/home/liangtao/Models/Qwen/Qwen2-0.5B-VocabPruned"
    temperature = 1.5
    args = {
        "model_name_or_path": "/home/liangtao/Models/Qwen/Qwen2-0.5B-VocabPruned",
        "finetuning_type": "lora",
        "template": "qwen",
        # "num_return_sequences": 5,
        # "infer_dtype": "float16",
        # "num_beams": 5,
        "temperature": temperature,
        # "temperature": 0,
        "max_new_tokens": 1000,
    }
    messages = [
        {"role": "user", "content": "What is the future of AI"}
    ]
    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
    engine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
    tokenizer = engine.tokenizer
    input_kwargs = {"num_return_sequences": 1, "output_scores": True, "return_dict_in_generate": True, "do_sample": True}
    gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
                engine.model,
                engine.tokenizer,
                None,
                engine.template,
                engine.generating_args,
                messages,
                input_kwargs=input_kwargs,
            )
    tokenizer.decode(gen_kwargs["inputs"][0], skip_special_tokens=False)
    sparse_model = SparseCacheModel(engine.model, window_size=512)
    output = sparse_model.generate(gen_kwargs)
    print(f"=======================\n{output}")