#!/bin/bash
# run_augmentation.sh

# 设置默认参数
DATA_DIR="/mnt/e/DataSets/Chemistry/RetroSynthesis"
DATA_FILE="retrosynthesis_train.json"
OUTPUT_DIR="/mnt/e/DataSets/Chemistry/RetroPrediction/HardExamples/train_augm"
VERSION="v1"
NUM_AUGMENTATIONS=3

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift
            shift
            ;;
        --data_file)
            DATA_FILE="$2"
            shift
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --version)
            VERSION="$2"
            shift
            shift
            ;;
        --num_augmentations)
            NUM_AUGMENTATIONS="$2"
            shift
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 执行Python脚本
python data_augmentation.py \
    --data_dir "$DATA_DIR" \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --version "$VERSION" \
    --num_augmentations "$NUM_AUGMENTATIONS"