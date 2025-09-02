import os

from llamafactory.chat import ChatModel
os.environ["TOKENIZERS_PARALLELISM"]="false"

INFER_ARGS = {
    "model_name_or_path": "/mnt/d/ChemicalFactory/output/internvl21_chemicals_retrosyn_full_para01",
    "finetuning_type": "full",
    "template": "intern_vl",
    # "infer_dtype": "float16",
    "num_beams": 5,
    "temperature": 0.95,
    "trust_remote_code": True,
    "max_new_tokens": 1000,
}

MESSAGES = [
    {"role": "user", "content": "<image>\nTRANSLATE_TO_SMILES:"},
]

def chat():
    chat_model = ChatModel(INFER_ARGS)
    images = ["/home/liangtao/Development/ChemistrySpace/ChemProphet/data/chemicals/retrosyn/images/bwatom/retrosyn_bwatom_00021579.png"]
    responses = chat_model.chat(messages=MESSAGES,
                                images=images,
                                num_return_sequences=5, output_scores=True, return_dict_in_generate=True,
                                do_sample=True,)
    print(responses)


def chat_batch():
    images1 = ["/home/liangtao/Development/ChemistrySpace/ChemProphet/data/chemicals/retrosyn/images/bwatom/retrosyn_bwatom_00021579.png"]
    images2 = ["/home/liangtao/Development/ChemistrySpace/ChemProphet/data/chemicals/retrosyn/images/bwatom/retrosyn_bwatom_00057555.png"]
    MESSAGES_LIST = []
    MESSAGES_LIST.append([{"role": "user", "images": images1, "content": "<image>\nTRANSLATE_TO_SMILES:"},])
    # MESSAGES_LIST.append([{"role": "user", "images": images2, "content": "<image>\nTRANSLATE_TO_SMILES:"},])
    chat_model = ChatModel(INFER_ARGS)
    responses = chat_model.batch_lmm_predict(MESSAGES_LIST, num_return_sequences=5, output_scores=True, return_dict_in_generate=True)
    print(responses)

if __name__=="__main__":
    # chat()
    chat_batch()