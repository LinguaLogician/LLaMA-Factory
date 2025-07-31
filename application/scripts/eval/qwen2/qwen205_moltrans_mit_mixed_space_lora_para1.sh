#!/bin/bash

MODEL_NAME="qwen205_moltrans_mit_mixed_space_lora_para1"
MODEL_PATH="/home/liangtao/Development/LLMSpace/LLaMA-Factory/output"
DATA_FILE="MIT_mixed.json"
FILE_PREFIX="mit_mixed_space_test_random100"
DATA_DIR="/home/liangtao/DataSets/Chemistry/MolecularTransformer/space/test/random100/"
PREDICTION_DIR="/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction/${FILE_PREFIX}/"

BATCH_LIMIT=1
BATCH_TOKEN_SIZE=400
MINMAX_GAP=20
NUM_RETURN_SEQUENCES=5
NUM_BEAMS=5


# 基础路径设置
BASE_DIR=$(dirname "$0")/../..
PREDICT_SCRIPT="./application/eval/qwen2/predict.py"
SCORE_SCRIPT="./application/eval/qwen2/score.py"

SCORE_OUTPUT_DIR="results/scores"
LOG_DIR="logs/predict"

# 创建必要的目录
mkdir -p "$LOG_DIR"
mkdir -p "$SCORE_OUTPUT_DIR"
mkdir -p "$PREDICTION_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 设置日志文件路径
LOG_FILE="${LOG_DIR}/${FILE_PREFIX}/${MODEL_NAME}_${TIMESTAMP}.log"

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