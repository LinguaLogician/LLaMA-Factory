from llamafactory.chat import ChatModel


INFER_ARGS = {
    "model_name_or_path": "/home/liangtao/Development/LLMSpace/LLaMA-Factory/output/qwen205_moltrans_mit_mixed_space_lora_para1_epoch3",
    "finetuning_type": "lora",
    "template": "qwen",
    # "num_return_sequences": 5,
    # "infer_dtype": "float16",
    "infer_backend": "sglang",
    "num_beams": 10,
    "temperature": 0,
    "max_new_tokens": 1000,
}

MESSAGES = [
    {"role": "user", "content": "PREDICT_PRODUCT:\nC 1 C C O C 1 . Cl . O C C C N 1 C C C C C 1 . O c 1 c c c ( - c 2 n c ( C N 3 C C C C C 3 ) c o 2 ) c c 1 F . c 1 c c c ( P ( c 2 c c c c c 2 ) c 2 c c c c c 2 ) c c 1"},
    # {"role": "user", "content": "PREDICT_PRODUCT:\nC . C C O C ( C ) = O . C c 1 o c ( - c 2 c c c c c 2 ) n c 1 C C O c 1 c c c ( [N+] ( = O ) [O-] ) c n 1 . [Pd]"},
]


def chat():
    chat_model = ChatModel(INFER_ARGS)
    response = chat_model.chat(MESSAGES)[0]
    print(response.response_text)


# Run tests if executed directly
if __name__ == "__main__":

    chat()
