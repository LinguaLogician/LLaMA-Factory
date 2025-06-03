
from llamafactory.chat import ChatModel

'''
generating_args: 
{
    "default_system": null,
    "do_sample": false,
    "enable_thinking": true,
    "length_penalty": 1,
    "max_new_tokens": 1000,
    "num_beams": 10,
    "repetition_penalty": 1,
    "skip_special_tokens": true,
    "temperature": 0.95,
    "top_k": 50,
    "top_p": 0.7
}
'''

INFER_ARGS = {
    # "model_name_or_path": "/home/liangtao/Models/Qwen/Qwen2-0.5B",
    "model_name_or_path": "/home/liangtao/Development/LLMSpace/LLaMA-Factory/output/qwen2_0_5b_molecular_transformer_mit_mixed_lora_sft_epoch_3",
    "finetuning_type": "lora",
    "template": "qwen",
    # "num_return_sequences": 5,
    # "infer_dtype": "float16",
    "num_beams": 10,
    "temperature": 0,
    # "temperature": 0,
    "max_new_tokens": 1000,
}

MESSAGES = [
    {"role": "user", "content": "PREDICT_PRODUCT:\nC 1 C C O C 1 . Cl . O C C C N 1 C C C C C 1 . O c 1 c c c ( - c 2 n c ( C N 3 C C C C C 3 ) c o 2 ) c c 1 F . c 1 c c c ( P ( c 2 c c c c c 2 ) c 2 c c c c c 2 ) c c 1"},
    # {"role": "user", "content": "PREDICT_PRODUCT:\nC . C C O C ( C ) = O . C c 1 o c ( - c 2 c c c c c 2 ) n c 1 C C O c 1 c c c ( [N+] ( = O ) [O-] ) c n 1 . [Pd]"},
]

def chat():
    chat_model = ChatModel(INFER_ARGS)
    responses = chat_model.chat(MESSAGES, num_return_sequences=10, output_scores=True, return_dict_in_generate=True,
                                do_sample=True,)
    # responses = chat_model.chat(MESSAGES, num_return_sequences=5)
    print(responses)


def chat_batch():
    MESSAGES_LIST = []
    MESSAGES_LIST.append([{"role": "user", "content": "PREDICT_PRODUCT:\nC 1 C C O C 1 . Cl . O C C C N 1 C C C C C 1 . O c 1 c c c ( - c 2 n c ( C N 3 C C C C C 3 ) c o 2 ) c c 1 F . c 1 c c c ( P ( c 2 c c c c c 2 ) c 2 c c c c c 2 ) c c 1"},])
    MESSAGES_LIST.append([{"role": "user", "content": "PREDICT_PRODUCT:\nC . C C O C ( C ) = O . C c 1 o c ( - c 2 c c c c c 2 ) n c 1 C C O c 1 c c c ( [N+] ( = O ) [O-] ) c n 1 . [Pd]"},])
    chat_model = ChatModel(INFER_ARGS)
    responses = chat_model.batch_llm_predict(MESSAGES_LIST, num_return_sequences=10, output_scores=True, return_dict_in_generate=True)
    print(responses)


# def stream_chat():
#     chat_model = ChatModel(INFER_ARGS)
#     response = ""
#     for token in chat_model.stream_chat(MESSAGES):
#         response += token



if __name__=="__main__":
    # chat()
    chat()