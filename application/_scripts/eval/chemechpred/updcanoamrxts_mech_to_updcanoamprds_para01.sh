#!/bin/bash
#https://chat.deepseek.com/a/chat/s/37c099f5-4e8f-4728-98e1-f6c613350785
# =============================================================================
# 常量设置区域（根据实际需求修改这些参数）
# =============================================================================

# 模型和任务参数
MODEL_PATH="/mnt/e/Development/LLMSpace/LLaMA-Factory/chemechpred/"
MODEL_NAME="updcanoamrxts_mech_to_updcanoamprds_para01"
TASK_ID="UPDCANOAMRXTS_MECH_TO_UPDCANOAMPRDS"

# 数据路径参数
DATA_DIR="/mnt/e/DataSets/Chemistry/ChemicalMechanism/via_random/test/"
PREDICTION_OUTPUT_DIR="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/chemechpred/prediction/"
SCORE_OUTPUT_DIR="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/chemechpred/scores/"

# Inference参数
FINETUNING_TYPE="full"
TEMPLATE="qwen"
NUM_BEAMS=5
DO_SAMPLE="true"
MAX_NEW_TOKENS=2048
NUM_RETURN_SEQUENCES=5
OUTPUT_SCORES="true"
RETURN_DICT_IN_GENERATE="true"

# 动态批处理参数
BATCH_LIMIT=4
BATCH_TOKEN_SIZE=1000
MINMAX_GAP=20

# GPU参数
GPU_THRESHOLD=20

# 日志参数
LOG_DIR="./logs/chemechpred"

# Python脚本路径
PREDICT_SCRIPT="application/eval/chemechpred/predict.py"
SCORE_SCRIPT="application/eval/chemechpred/score.py"

# =============================================================================
# 执行区域（通常不需要修改）
# =============================================================================

# 创建必要的目录
mkdir -p "$LOG_DIR"
mkdir -p "$PREDICTION_OUTPUT_DIR"
mkdir -p "$SCORE_OUTPUT_DIR"

echo "开始执行化学机制预测评估..."
echo "当前时间: $(date)"
echo "=========================================="

# 执行预测脚本
echo "执行预测阶段..."
python "$PREDICT_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --task_id "$TASK_ID" \
    --data_dir "$DATA_DIR" \
    --output_dir "$PREDICTION_OUTPUT_DIR" \
    --finetuning_type "$FINETUNING_TYPE" \
    --template "$TEMPLATE" \
    --num_beams "$NUM_BEAMS" \
    $( [ "$DO_SAMPLE" = "true" ] && echo "--do_sample" ) \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    $( [ "$OUTPUT_SCORES" = "true" ] && echo "--output_scores" ) \
    $( [ "$RETURN_DICT_IN_GENERATE" = "true" ] && echo "--return_dict_in_generate" ) \
    --batch_limit "$BATCH_LIMIT" \
    --batch_token_size "$BATCH_TOKEN_SIZE" \
    --minmax_gap "$MINMAX_GAP" \
    --gpu_threshold "$GPU_THRESHOLD" \

if [ $? -eq 0 ]; then
    echo "预测阶段完成！"
else
    echo "预测阶段失败！"
    exit 1
fi

echo "=========================================="

# 执行评分脚本
echo "执行评分阶段..."
TASK_ID_LOWER=$(echo "$TASK_ID" | tr '[:upper:]' '[:lower:]')
PREDICTION_FILE_PATH="${PREDICTION_OUTPUT_DIR}/${TASK_ID_LOWER}/${MODEL_NAME}.json"

python "$SCORE_SCRIPT" \
    --prediction_dir "$(dirname "$PREDICTION_FILE_PATH")" \
    --prediction_file "$(basename "$PREDICTION_FILE_PATH" .json)" \
    --output_dir "${SCORE_OUTPUT_DIR}/${TASK_ID_LOWER}"

if [ $? -eq 0 ]; then
    echo "评分阶段完成！"
else
    echo "评分阶段失败！"
    exit 1
fi

echo "=========================================="
echo "化学机制预测评估全部完成！"
echo "完成时间: $(date)"
echo "预测结果: $PREDICTION_FILE_PATH"
echo "评分结果: ${SCORE_OUTPUT_DIR}/${TASK_ID_LOWER}/$(basename "$PREDICTION_FILE_PATH" .json)_scores.json"