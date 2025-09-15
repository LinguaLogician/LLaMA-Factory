#!/bin/bash

# Parameters
MIN_MEMORY=15000  # Minimum required GPU memory in MB (adjust as needed)
POLL_INTERVAL=600  # Time interval between checks in seconds

PARAM_PATH="examples/train_full/qwen205/nospace/qwen205_retrosyn_nospace_full_para6.yaml"  # Example path

# Get current timestamp for log filename
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PARAM_BASENAME=$(basename "$PARAM_PATH" .yaml)
LOG_FILE_NAME="${TIMESTAMP}.log"
LOG_DIR="logs/train/${PARAM_BASENAME}"
LOG_FILE_PATH="${LOG_DIR}/${LOG_FILE_NAME}"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to get available GPU memory in MB
get_available_gpu_memory() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | awk '{print $1}'
}

# Function to log messages to both console and log file
log_message() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE_PATH"
}

# Main execution
log_message "Starting training script with parameters:"
log_message "  PARAM_PATH: $PARAM_PATH"
log_message "  MIN_MEMORY: ${MIN_MEMORY}MB"
log_message "  POLL_INTERVAL: ${POLL_INTERVAL}s"
log_message "  LOG_DIR: $LOG_DIR"

while true; do
    AVAILABLE_MEM=$(get_available_gpu_memory)

    if [ "$AVAILABLE_MEM" -ge "$MIN_MEMORY" ]; then
        log_message "Sufficient GPU memory available (${AVAILABLE_MEM}MB >= ${MIN_MEMORY}MB). Starting training..."
        log_message "Executing: llamafactory-cli train $PARAM_PATH"

        # Execute the command and capture all output to the log file
        FORCE_TORCHRUN=1 llamafactory-cli train "$PARAM_PATH" >> "$LOG_FILE_PATH" 2>&1

        TRAIN_EXIT_CODE=$?
        if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
            log_message "Training completed successfully."
        else
            log_message "Training failed with exit code $TRAIN_EXIT_CODE."
        fi
        break
    else
        log_message "Insufficient GPU memory (${AVAILABLE_MEM}MB < ${MIN_MEMORY}MB). Waiting..."
        sleep "$POLL_INTERVAL"
    fi
done

log_message "Script completed."