#!/bin/bash

# 定义路径
model_path="/home/liangtao/Development/LLMSpace/LLaMA-Factory/output/qwen205_moltrans_stereo_mixed_nospace_full_para1"
script_path="application/scripts/eval/qwen2/qwen205_moltrans_stereo_mixed_nospace_full_para1_test1.sh"

# 目标检查点
target_checkpoint="checkpoint-331698"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "开始监控检查点 $target_checkpoint..."

while true; do
    # 检查目标文件夹是否存在
    if [ -d "$model_path/$target_checkpoint" ]; then
        log "发现检查点 $target_checkpoint，准备执行脚本..."

        # 检查脚本是否存在并可执行
        if [ -f "$script_path" ] && [ -x "$script_path" ]; then
            log "正在执行脚本: $script_path"
            # 执行脚本并捕获输出
            if bash "$script_path"; then
                log "脚本执行成功"
                break  # 执行成功后退出循环
            else
                log "脚本执行失败，退出状态码: $?"
                break
            fi
        else
            log "错误: 脚本不存在或不可执行"
            break
        fi
    else
        log "检查点 $target_checkpoint 尚未出现，继续监控..."
    fi

    # 等待5分钟
    sleep 300
done

log "监控任务结束"