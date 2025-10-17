#!/bin/bash

# 批量运行化学机制预测和评分脚本
# 使用方法: ./batch_run.sh

# 设置基础目录
BASE_DIR="$(pwd)"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 任务配置数组
# 格式: "配置文件路径:额外参数"
TASKS=(
    # 多任务配置
    "application/chemech/eval/_config/multi_task/enhc_rxts_to_prds_v4_1_and_v5_1.json:--batch_limit 3"
    # 单任务配置
    "application/chemech/eval/_config/single_task/rxts_to_mech_v1.json:--batch_limit 3"
)

# 运行所有任务
echo "开始批量运行 ${#TASKS[@]} 个任务..."
echo "日志目录: $LOG_DIR"
echo "=========================================="

for i in "${!TASKS[@]}"; do
    # 解析任务配置
    IFS=':' read -r CONFIG_FILE EXTRA_ARGS <<< "${TASKS[$i]}"

    # 检查配置文件是否存在
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "警告: 配置文件 $CONFIG_FILE 不存在，跳过任务 $((i+1))"
        continue
    fi

    # 生成任务名称用于日志文件
    TASK_NAME=$(basename "$CONFIG_FILE" .json)
    LOG_FILE="${LOG_DIR}/${TASK_NAME}_${TIMESTAMP}.log"

    echo "运行任务 $((i+1))/${#TASKS[@]}: $CONFIG_FILE"
    echo "日志文件: $LOG_FILE"
    echo "额外参数: $EXTRA_ARGS"

#    # 运行Python脚本 - 实时输出到控制台和日志文件
#    python "${BASE_DIR}/application/chemech/eval/batch_predict_score.py" \
#        --config_file "$CONFIG_FILE" \
#        --output_base_dir "results/chemechpred/prediction_${TASK_NAME}" \
#        --score_base_dir "results/chemechpred/scores_${TASK_NAME}" \
#        $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"
#    # 修改这一行：
    python -u "${BASE_DIR}/application/chemech/eval/batch_predict_score.py" \
      --config_file "$CONFIG_FILE" \
      --output_base_dir "results/chemechpred/prediction_${TASK_NAME}" \
      --score_base_dir "results/chemechpred/scores_${TASK_NAME}" \
      $EXTRA_ARGS 2>&1 | stdbuf -o0 tee "$LOG_FILE"
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 任务 $((i+1)) 完成"
    else
        echo "✗ 任务 $((i+1)) 失败，查看日志: $LOG_FILE"
    fi

    echo "------------------------------------------"

    # 任务间延迟，避免资源冲突
    if [ $i -lt $((${#TASKS[@]} - 1)) ]; then
        echo "等待30秒后执行下一个任务..."
        sleep 30
    fi
done

echo "批量运行完成!"
echo "查看日志文件: $LOG_DIR"