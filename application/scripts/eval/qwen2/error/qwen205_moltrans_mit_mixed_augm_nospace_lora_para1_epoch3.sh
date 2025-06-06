#!/bin/bash

# ./application/scripts/eval/qwen2/qwen205_moltrans_mit_mixed_augm_nospace_lora_para1_epoch3.yaml

# 设置默认参数
DEFAULT_MODEL_NAME="qwen205_moltrans_mit_mixed_augm_nospace_lora_para1_epoch3"
DEFAULT_DATA_FILE="MIT_mixed_augm.json"
DEFAULT_BATCH_LIMIT=1
DEFAULT_BATCH_TOKEN_SIZE=400
DEFAULT_MINMAX_GAP=20
DEFAULT_NUM_RETURN_SEQUENCES=5
DEFAULT_NUM_BEAMS=5
FILE_PREFIX="mit_mixed_augm_space_test"
# 基础路径设置
BASE_DIR=$(dirname "$0")/../..
PREDICT_SCRIPT="./application/eval/qwen2/predict.py"
SCORE_SCRIPT="./application/eval/qwen2/score.py"
DATA_DIR="/home/liangtao/DataSets/Chemistry/MolecularTransformer/space/test/"
MODEL_PATH="/home/liangtao/Development/LLMSpace/LLaMA-Factory/output"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --batch_limit)
            BATCH_LIMIT="$2"
            shift 2
            ;;
        --batch_token_size)
            BATCH_TOKEN_SIZE="$2"
            shift 2
            ;;
        --minmax_gap)
            MINMAX_GAP="$2"
            shift 2
            ;;
        --num_return_sequences)
            NUM_RETURN_SEQUENCES="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --prediction_dir)
            PREDICTION_DIR="$2"
            shift 2
            ;;
        --score_output_dir)
            SCORE_OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 设置默认值
MODEL_NAME=${MODEL_NAME:-$DEFAULT_MODEL_NAME}
DATA_FILE=${DATA_FILE:-$DEFAULT_DATA_FILE}
BATCH_LIMIT=${BATCH_LIMIT:-$DEFAULT_BATCH_LIMIT}
BATCH_TOKEN_SIZE=${BATCH_TOKEN_SIZE:-$DEFAULT_BATCH_TOKEN_SIZE}
MINMAX_GAP=${MINMAX_GAP:-$DEFAULT_MINMAX_GAP}
NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES:-$DEFAULT_NUM_RETURN_SEQUENCES}
NUM_BEAMS=${NUM_BEAMS:-$DEFAULT_NUM_BEAMS}
LOG_DIR=${LOG_DIR:-"logs/predict"}
PREDICTION_DIR=${PREDICTION_DIR:-"/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction"}
SCORE_OUTPUT_DIR=${SCORE_OUTPUT_DIR:-"results/scores"}

# 创建日志目录
mkdir -p "$LOG_DIR"
mkdir -p "$SCORE_OUTPUT_DIR"

# 设置日志文件路径
LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${FILE_PREFIX}.log"

# 清空或创建日志文件
> "$LOG_FILE"

# 执行预测脚本并记录日志
echo "Starting prediction at $(date)" | tee -a "$LOG_FILE"
python "$PREDICT_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$PREDICTION_DIR" \
    --finetuning_type "lora" \
    --template "qwen" \
    --num_beams "$NUM_BEAMS" \
    --max_new_tokens 1000 \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --batch_limit "$BATCH_LIMIT" \
    --batch_token_size "$BATCH_TOKEN_SIZE" \
    --minmax_gap "$MINMAX_GAP" \
    2>&1 | tee -a "$LOG_FILE"

PREDICTION_EXIT_CODE=${PIPESTATUS[0]}

if [ $PREDICTION_EXIT_CODE -ne 0 ]; then
    echo "Prediction failed with exit code $PREDICTION_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $PREDICTION_EXIT_CODE
fi

echo "Prediction completed successfully at $(date)" | tee -a "$LOG_FILE"

# 执行评分脚本
PREDICTION_FILE="${MODEL_NAME}"  # 不含.json后缀

echo "Starting scoring at $(date)" | tee -a "$LOG_FILE"
python "$SCORE_SCRIPT" \
    --prediction_file "$PREDICTION_FILE" \
    --prediction_dir "$PREDICTION_DIR" \
    --output_dir "$SCORE_OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

SCORE_EXIT_CODE=${PIPESTATUS[0]}

if [ $SCORE_EXIT_CODE -ne 0 ]; then
    echo "Scoring failed with exit code $SCORE_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $SCORE_EXIT_CODE
fi

echo "Scoring completed successfully at $(date)" | tee -a "$LOG_FILE"
echo "All tasks completed" | tee -a "$LOG_FILE"