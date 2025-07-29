import os
import json

from tqdm import tqdm


def convert_to_json(base_dir, output_dir, task, nospace=False, split="train"):

    if os.path.isdir(base_dir):
        data_file = os.path.join(base_dir, f"retrosynthesis-{split}.smi")

        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as src_f:
                rxn_lines = src_f.readlines()

            json_data = []
            for idx, rxn_line in enumerate(tqdm(rxn_lines, desc="Processing")):
                (src_line, tgt_line) = rxn_line.split(">>")
                entry = {
                    "id": f"retrosynthesis_{split}_{str(idx + 1).zfill(9)}",
                    "instruction": f"{task}:",
                    "input": src_line.strip().replace(' ', '') if nospace else src_line.strip(),
                    "output": tgt_line.strip().replace(' ', '') if nospace else tgt_line.strip()
                }
                json_data.append(entry)

            output_folder = os.path.join(output_dir)
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(str(output_folder), f"retrosynthesis_{split}.json")
            with open(output_file, 'w', encoding='utf-8') as json_f:
                json.dump(json_data, json_f, indent=2, ensure_ascii=False)

            print(f"Successfully created retrosynthesis_{split}.json")


if __name__=='__main__':
    base_dir = r"/home/liangtao/Development/ChemistrySpace/retrosynthesis/data"
    output_dir = r"/home/liangtao/DataSets/Chemistry/RetroSynthesis"
    task = "PREDICT_REACTANTS"
    convert_to_json(base_dir, output_dir, task, nospace=True, split="train")
    convert_to_json(base_dir, output_dir, task, nospace=True, split="valid")
    convert_to_json(base_dir, output_dir, task, nospace=True, split="test")