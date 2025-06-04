import os
import json

from tqdm import tqdm


def convert_to_json(base_dir, output_dir, task, nospace=False, split="train"):
    for sub_folder in os.listdir(base_dir):
        sub_folder_path = os.path.join(base_dir, sub_folder)

        if os.path.isdir(sub_folder_path):
            src_file = os.path.join(sub_folder_path, f"src-{split}.txt")
            tgt_file = os.path.join(sub_folder_path, f"tgt-{split}.txt")

            if os.path.exists(src_file) and os.path.exists(tgt_file):
                with open(src_file, 'r', encoding='utf-8') as src_f, \
                        open(tgt_file, 'r', encoding='utf-8') as tgt_f:
                    src_lines = src_f.readlines()
                    tgt_lines = tgt_f.readlines()

                if len(src_lines) != len(tgt_lines):
                    print(f"Warning: {sub_folder} has mismatched line counts in src-train.txt and tgt-train.txt")
                    continue

                json_data = []
                format = "nospace" if nospace else "space"
                for idx, (src_line, tgt_line) in enumerate(tqdm(zip(src_lines, tgt_lines), desc="Processing")):
                    entry = {
                        "id": f"{sub_folder}_{split}_{format}_{str(idx + 1).zfill(9)}",
                        "instruction": f"{task}:",
                        "input": src_line.strip().replace(' ', '') if nospace else src_line.strip(),
                        "output": tgt_line.strip().replace(' ', '') if nospace else tgt_line.strip()
                    }
                    json_data.append(entry)

                output_folder = os.path.join(output_dir, format, split)
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, f"{sub_folder}.json")
                with open(output_file, 'w', encoding='utf-8') as json_f:
                    json.dump(json_data, json_f, indent=2, ensure_ascii=False)

                print(f"Successfully created {sub_folder}.json")


if __name__=='__main__':
    base_dir = r"/home/liangtao/Development/ChemistrySpace/MolecularTransformer/data"
    output_dir = r"/home/liangtao/DataSets/Chemistry/MolecularTransformer"
    task = "PREDICT_PRODUCT"
    convert_to_json(base_dir, output_dir, task, nospace=True, split="train")
    convert_to_json(base_dir, output_dir, task, nospace=True, split="val")
    convert_to_json(base_dir, output_dir, task, nospace=True, split="test")
    convert_to_json(base_dir, output_dir, task, nospace=False, split="train")
    convert_to_json(base_dir, output_dir, task, nospace=False, split="val")
    convert_to_json(base_dir, output_dir, task, nospace=False, split="test")

    # python ./application/data/moltrans/convert.py