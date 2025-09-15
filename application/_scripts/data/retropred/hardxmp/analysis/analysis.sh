#!/bin/bash

# 默认参数
DATA_DIR="/mnt/e/Development/LLMSpace/LLaMA-Factory/results/prediction/retrosyn_nospace_test"
DATA_FILE="qwen205_retrosyn_nospace_full_para1_ckptlast.json"
TARGET_DIR="/mnt/e/DataSets/Chemistry/RetroPrediction/HardExamples/prediction_test/qwen205_retrosyn_nospace_full_para1_ckptlast"
HARDNESS_THRESHOLDS=(0.3 0.6)

# 显示用法信息
usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -d, --data_dir DIR         数据目录路径 (默认: $DATA_DIR)"
    echo "  -f, --data_file FILE       数据文件名 (默认: $DATA_FILE)"
    echo "  -t, --target_dir DIR       目标目录路径 (默认: $TARGET_DIR)"
    echo "  -h, --hardness FLOATS      困难度阈值列表，用空格分隔 (默认: ${HARDNESS_THRESHOLDS[*]})"
    echo "  --help                     显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0"
    echo "  $0 -d /path/to/data -f data.json -t /path/to/target -h 0.2 0.5 0.8"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -f|--data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        -t|--target_dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        -h|--hardness)
            shift
            HARDNESS_THRESHOLDS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                HARDNESS_THRESHOLDS+=("$1")
                shift
            done
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            usage
            exit 1
            ;;
    esac
done

# 检查Python脚本是否存在
PYTHON_SCRIPT="process_hardness.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "错误: Python脚本 '$PYTHON_SCRIPT' 不存在"
    echo "请确保脚本与Shell脚本在同一目录下"
    exit 1
fi

# 检查数据文件是否存在
DATA_PATH="$DATA_DIR/$DATA_FILE"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "错误: 数据文件 '$DATA_PATH' 不存在"
    exit 1
fi

# 将阈值数组转换为字符串
THRESHOLDS_STR=$(printf "%s " "${HARDNESS_THRESHOLDS[@]}" | sed 's/ $//')

# 显示配置信息
echo "=========================================="
echo "          数据处理脚本配置"
echo "=========================================="
echo "数据目录: $DATA_DIR"
echo "数据文件: $DATA_FILE"
echo "目标目录: $TARGET_DIR"
echo "困难度阈值: ${THRESHOLDS_STR}"
echo "Python脚本: $PYTHON_SCRIPT"
echo "=========================================="
echo ""

# 确认执行
read -p "是否开始执行? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

echo "开始执行..."

# 执行Python脚本
python "$PYTHON_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --data_file "$DATA_FILE" \
    --target_dir "$TARGET_DIR" \
    --hardness_thresholds "${HARDNESS_THRESHOLDS[@]}"

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "          处理完成!"
    echo "=========================================="
    echo "输出文件位于: $TARGET_DIR"
    echo "摘要文件: ${DATA_FILE%.*}_summary.txt"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "          处理失败!"
    echo "=========================================="
    exit 1
fi