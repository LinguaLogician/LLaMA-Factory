#!/bin/bash

# Configuration section
SCRIPT_PATH="application/model/save/merge.py"
LOG_DIR="logs/merge"
PROJECT_PATH="/home/liangtao/Development/LLMSpace/LLaMA-Factory"


# Set 1 Parameters
PARAM_SET_1=(
# yaml_relative_path
  "examples/merge_lora/qwen205/common.yaml"
#  adapter_path
  "saves/sft_lora/qwen205_moltrans_mit_mixed_nospace_lora_para3"
#  output_dir
  "output/qwen205_moltrans_mit_mixed_nospace_lora_para3"
#  merge_interval
  "20000"
#  save_last
  ""
#  regular_save
  ""
#  start_ckpt
  ""
#  extra_checkpoints
#  "200000"
  ""
#  threads
  "4"
)

# Add more parameter sets as needed
# PARAM_SET_3=(...)
# PARAM_SET_4=(...)

# Collect all parameter sets
ALL_PARAM_SETS=(
  "${PARAM_SET_1[@]}"
#  "${PARAM_SET_2[@]}"
  # Add more sets here
  # "${PARAM_SET_3[@]}"
  # "${PARAM_SET_4[@]}"
)

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/merge_${TIMESTAMP}.log"

# Function to run a single parameter set
run_param_set() {
  local params=("$@")
  local yaml_relative_path="${params[0]}"
  local adapter_path="${params[1]}"
  local output_dir="${params[2]}"
  local merge_interval="${params[3]}"
  local save_last="${params[4]}"
  local regular_save="${params[5]}"
  local start_ckpt="${params[6]}"
  local extra_checkpoints="${params[7]}"
  local threads="${params[8]}"

  # Build command
  local cmd="python $SCRIPT_PATH \
    --yaml_relative_path \"$yaml_relative_path\" \
    --project_path \"$PROJECT_PATH\" \
    --adapter_path \"$adapter_path\" \
    --output_dir \"$output_dir\" \
    --merge_interval \"$merge_interval\""

  # Add optional boolean flags
#  [ "$save_last" = "1" ] && cmd+=" --save_last"
  [ "$save_last" = "1" ] && cmd+=" --save_last 1"
#  [ "$regular_save" = "1" ] && cmd+=" --regular_save"
  [ "$regular_save" = "1" ] && cmd+=" --regular_save 1"


  # Add optional parameters
  [ -n "$start_ckpt" ] && cmd+=" --start_ckpt \"$start_ckpt\""
  # 如果 start_ckpt 为空，则设置为 0
#  [ -n "$start_ckpt" ] && cmd+=" --start_ckpt \"$start_ckpt\"" || cmd+=" --start_ckpt 0"
  [ -n "$extra_checkpoints" ] && cmd+=" --extra_checkpoints $extra_checkpoints"

  cmd+=" --threads \"$threads\""

  # Log the command
  echo "======================================================================" | tee -a "$LOG_FILE"
  echo "Executing command: $cmd" | tee -a "$LOG_FILE"
  echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
  echo "----------------------------------------------------------------------" | tee -a "$LOG_FILE"

  # Execute the command and capture all output
  eval "$cmd" 2>&1 | tee -a "$LOG_FILE"

  # Capture exit status
  local status=${PIPESTATUS[0]}

  echo "----------------------------------------------------------------------" | tee -a "$LOG_FILE"
  echo "Command completed with exit status: $status" | tee -a "$LOG_FILE"
  echo "======================================================================" | tee -a "$LOG_FILE"

  return $status
}

# Process all parameter sets
TOTAL_SETS=$((${#ALL_PARAM_SETS[@]} / 9))
for (( i=0; i<$TOTAL_SETS; i++ )); do
  OFFSET=$(( i * 9 ))
  CURRENT_SET=("${ALL_PARAM_SETS[@]:$OFFSET:9}")

  echo "Processing parameter set $((i+1)) of $TOTAL_SETS" | tee -a "$LOG_FILE"
  run_param_set "${CURRENT_SET[@]}"
  STATUS=$?

  if [ $STATUS -ne 0 ]; then
    echo "Warning: Parameter set $((i+1)) failed with status $STATUS" | tee -a "$LOG_FILE"
  fi
done

echo "All parameter sets processed. Log file: $LOG_FILE" | tee -a "$LOG_FILE"