#!/bin/bash

MODEL_NAME="qwen205_retrosyn_nospace_full_para10"
MODEL_PATH="/home/vipuser/Development/LLMSpace/LLaMA-Factory/output"
DATA_FILE="retrosynthesis_test.json"
CHECKPOINTS=""
FILE_PREFIX="retrosyn_nospace_test"
DATA_DIR="/home/vipuser/DataSets/Chemistry/RetroSynthesis/"
PREDICTION_BASE_DIR="/home/vipuser/Development/LLMSpace/LLaMA-Factory/results/prediction/${FILE_PREFIX}/"

BATCH_LIMIT=1
BATCH_TOKEN_SIZE=600
MINMAX_GAP=20
NUM_RETURN_SEQUENCES=5
NUM_BEAMS=5

# Base path settings
BASE_DIR=$(dirname "$0")/../..
PREDICT_SCRIPT="./application/eval/qwen2/predict.py"
SCORE_SCRIPT="./application/eval/qwen2/score.py"

SCORE_OUTPUT_DIR="results/scores"
LOG_DIR="logs/predict"

# Create necessary directories
mkdir -p "$LOG_DIR/${FILE_PREFIX}"
mkdir -p "$SCORE_OUTPUT_DIR"
mkdir -p "$PREDICTION_BASE_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to process each checkpoint
process_checkpoint() {
    local checkpoint=$1
    local prediction_dir="${PREDICTION_BASE_DIR}"

    mkdir -p "$prediction_dir"
    mkdir -p "${LOG_DIR}/${FILE_PREFIX}"
    local log_file="${LOG_DIR}/${FILE_PREFIX}/${MODEL_NAME}_${checkpoint}_${TIMESTAMP}.log"
    > "$log_file"

    echo "Starting prediction for checkpoint $checkpoint at $(date)" | tee -a "$log_file"

    # Run prediction
    python "$PREDICT_SCRIPT" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --data_file "$DATA_FILE" \
        --checkpoints "$checkpoint" \
        --data_dir "$DATA_DIR" \
        --output_dir "$prediction_dir" \
        --finetuning_type "lora" \
        --template "qwen" \
        --num_beams "$NUM_BEAMS" \
        --max_new_tokens 1000 \
        --num_return_sequences "$NUM_RETURN_SEQUENCES" \
        --batch_limit "$BATCH_LIMIT" \
        --batch_token_size "$BATCH_TOKEN_SIZE" \
        --minmax_gap "$MINMAX_GAP" \
        2>&1 | tee -a "$log_file"

    local prediction_exit_code=${PIPESTATUS[0]}

    if [ $prediction_exit_code -ne 0 ]; then
        echo "Prediction failed for checkpoint $checkpoint with exit code $prediction_exit_code" | tee -a "$log_file"
        return $prediction_exit_code
    fi

    echo "Prediction completed successfully for checkpoint $checkpoint at $(date)" | tee -a "$log_file"

    # Run scoring
    echo "Starting scoring for checkpoint $checkpoint at $(date)" | tee -a "$log_file"

    python "$SCORE_SCRIPT" \
        --prediction_file "${MODEL_NAME}_ckptlast" \
        --prediction_dir "$prediction_dir" \
        --output_dir "$SCORE_OUTPUT_DIR" \
        2>&1 | tee -a "$log_file"

    local score_exit_code=${PIPESTATUS[0]}

    if [ $score_exit_code -ne 0 ]; then
        echo "Scoring failed for checkpoint $checkpoint with exit code $score_exit_code" | tee -a "$log_file"
        return $score_exit_code
    fi

    echo "Scoring completed successfully for checkpoint $checkpoint at $(date)" | tee -a "$log_file"
    return 0
}

# Process each checkpoint
process_checkpoint ""
if [ $? -ne 0 ]; then
    echo "Error processing checkpoint $checkpoint, exiting..." | tee -a "$log_file"
    exit 1
fi

echo "All tasks completed successfully at $(date)" | tee -a "$log_file"