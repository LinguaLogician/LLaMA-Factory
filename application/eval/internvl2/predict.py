# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: predict.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/2 10:13
# https://chat.deepseek.com/a/chat/s/cb883b87-3991-4c5f-99c9-dcbd86eb2fed

import os
import json
import argparse
from tqdm import tqdm
from rdkit import Chem
from llamafactory.chat import ChatModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid using RDKit."""
    try:
        # Remove any spaces before checking
        smiles = smiles.strip().replace(" ", "")
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def are_same_compound(smiles1: str, smiles2: str) -> bool:
    """Check if two SMILES strings represent the same compound using RDKit."""
    try:
        # Remove spaces and standardize
        smiles1 = smiles1.strip().replace(" ", "")
        smiles2 = smiles2.strip().replace(" ", "")

        if not is_valid_smiles(smiles1) or not is_valid_smiles(smiles2):
            return False

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return False

        # Compare canonical SMILES
        can_smiles1 = Chem.MolToSmiles(mol1, canonical=True)
        can_smiles2 = Chem.MolToSmiles(mol2, canonical=True)

        return can_smiles1 == can_smiles2
    except:
        return False


def process_data(args):
    """Main function to process the data and generate predictions."""

    # Load test data
    test_file_path = os.path.join(args.test_path, args.test_file)
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    print(f"Loaded {len(test_data)} test samples from {test_file_path}")

    # Initialize model
    model_path = os.path.join(args.model_location, args.model_name)
    infer_args = {
        "model_name_or_path": model_path,
        "finetuning_type": args.finetuning_type,
        "template": args.template,
        "num_beams": args.num_beams,
        "temperature": args.temperature,
        "trust_remote_code": args.trust_remote_code,
        "max_new_tokens": args.max_new_tokens,
    }

    if args.infer_dtype:
        infer_args["infer_dtype"] = args.infer_dtype

    chat_model = ChatModel(infer_args)

    results = []

    # Process each test sample
    for item in tqdm(test_data, desc="Processing test samples"):
        # Extract data from test item
        item_id = item["id"]
        image_path = item["image"]

        # Extract instruction and label from conversations
        instruction = ""
        label = ""
        for conv in item["conversations"]:
            if conv["from"] == "human":
                instruction = conv["value"]
            elif conv["from"] == "gpt":
                label = conv["value"]

        # Prepare messages for model
        messages = [{"role": "user", "content": instruction}]

        # Get model responses
        try:
            responses = chat_model.chat(
                messages=messages,
                images=[image_path],
                num_return_sequences=args.num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=args.do_sample,
            )
        except Exception as e:
            print(f"Error processing {item_id}: {e}")
            continue

        # Process responses
        output_list = []
        # Sort responses by sequence_score in descending order
        sorted_responses = sorted(responses, key=lambda x: x.sequence_score, reverse=True)

        for resp in sorted_responses:
            output_item = {
                "text": resp.response_text,
                "length": resp.response_length,
                "sequence_score": resp.sequence_score,
                "is_valid": is_valid_smiles(resp.response_text),
                "is_correct": are_same_compound(resp.response_text, label)
            }
            output_list.append(output_item)

        # Create result entry
        result_entry = {
            "id": item_id,
            "instruction": instruction,
            "input": "",
            "label": label,
            "prompt_length": responses[0].prompt_length if responses else 0,
            "output": output_list
        }
        results.append(result_entry)

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Process model inference for chemical SMILES generation")

    # Model parameters
    parser.add_argument("--model_location", type=str, default="/mnt/d/ChemicalFactory/output/",
                        help="Path to the model directory")
    parser.add_argument("--model_name", type=str, default="internvl21_chemicals_retrosyn_full_para01",
                        help="Name of the model")

    # Test data parameters
    parser.add_argument("--test_path", type=str,
                        default="/mnt/e/Development/ChemistrySpace/ChemProphet/data/chemicals/retrosyn/",
                        help="Path to the test data directory")
    parser.add_argument("--test_file", type=str, default="retrosyn_test_internvl2.jsonl",
                        help="Name of the test file")

    # Inference parameters
    parser.add_argument("--finetuning_type", type=str, default="full",
                        help="Finetuning type")
    parser.add_argument("--template", type=str, default="intern_vl",
                        help="Template for the model")
    parser.add_argument("--infer_dtype", type=str, default=None,
                        help="Inference data type")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.95,
                        help="Temperature for sampling")
    parser.add_argument("--trust_remote_code", type=bool, default=True,
                        help="Whether to trust remote code")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_return_sequences", type=int, default=5,
                        help="Number of sequences to return")
    parser.add_argument("--do_sample", type=bool, default=True,
                        help="Whether to use sampling")

    # Output parameters
    parser.add_argument("--output_dir", type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction/chemicals_retrosyn_test",
                        help="Directory to save output results")

    args = parser.parse_args()

    # Process the data
    process_data(args)


if __name__ == "__main__":
    main()
