import os

from llamafactory.chat import ChatModel
os.environ["TOKENIZERS_PARALLELISM"]="false"

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
    "model_name_or_path": "/mnt/d/ChemicalFactory/output/internvl21_chemicals_retrosyn_full_para01",
    "finetuning_type": "full",
    "template": "intern_vl",
    # "infer_dtype": "float16",
    "num_beams": 5,
    "temperature": 0.95,
    "trust_remote_code": True,
    "local_files_only": True,
    # "temperature": 0,
    "max_new_tokens": 1000,
}

MESSAGES = [
    # {"role": "user", "content": "What is the future of AI"}
    # {"role": "user", "content": "PREDICT_PRODUCT:\nC I . N N 1 C C O C C 1"},
    {"role": "user", "content": "<image>\nTRANSLATE_TO_SMILES:"},
    # {"role": "user", "content": "PREDICT_PRODUCT:\nC . C C O C ( C ) = O . C c 1 o c ( - c 2 c c c c c 2 ) n c 1 C C O c 1 c c c ( [N+] ( = O ) [O-] ) c n 1 . [Pd]"},
]

def chat():
    # torch.manual_seed(int(time.time()))
    chat_model = ChatModel(INFER_ARGS)
    images = ["/home/liangtao/Development/ChemistrySpace/ChemProphet/data/chemicals/retrosyn/images/bwatom/retrosyn_bwatom_00021579.png"]
    responses = chat_model.chat(messages=MESSAGES,
                                images=images,
                                num_return_sequences=5, output_scores=True, return_dict_in_generate=True,
                                do_sample=True,)
    # responses = chat_model.chat(MESSAGES, num_return_sequences=5)
    print(responses)


def chat_batch():
    MESSAGES_LIST = []
    # MESSAGES_LIST.append([{"role": "user", "content": "PREDICT_PRODUCT:\nC 1 C C O C 1 . Cl . O C C C N 1 C C C C C 1 . O c 1 c c c ( - c 2 n c ( C N 3 C C C C C 3 ) c o 2 ) c c 1 F . c 1 c c c ( P ( c 2 c c c c c 2 ) c 2 c c c c c 2 ) c c 1"},])
    MESSAGES_LIST.append([{"role": "user", "content": "What is the future of AI"},])
    chat_model = ChatModel(INFER_ARGS)
    responses = chat_model.batch_llm_predict(MESSAGES_LIST, num_return_sequences=10, output_scores=True, return_dict_in_generate=True)
    print(responses)


# def stream_chat():
#     chat_model = ChatModel(INFER_ARGS)
#     response = ""
#     for token in chat_model.stream_chat(MESSAGES):
#         response += token



if __name__=="__main__":
    chat()