#!/bin/bash

# 设置参数
BASE_DIR="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/prediction/mit_mixed_nospace_test"
DATA_FILE="qwen2505_moltrans_mit_mixed_nospace_full_para01_ckptlast.json"
OUTPUT_DIR="/mnt/e/DataSets/Chemistry/ForwardPrediction/RewardModel/mit_mixed/{split}/"

# 执行Python脚本
python application/data/fwdprediction/rlhf/data_split.py \
    --base_dir "$BASE_DIR" \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio 0.8 \
    --valid_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42