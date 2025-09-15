#!/bin/bash

# 设置参数
BASE_DIR="/mnt/e/DataSets/Chemistry/MolecularTransformer/nospace/train"
DATA_FILE="MIT_mixed.json"
OUTPUT_DIR="/mnt/e/DataSets/Chemistry/ForwardPrediction/RLHF/{split}/"

# 执行Python脚本
python application/data/fwdprediction/rlhf/train_data_split.py \
    --base_dir "$BASE_DIR" \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --sft_ratio 0.4 \
    --reward_ratio 0.3 \
    --ppo_ratio 0.3 \
    --seed 42