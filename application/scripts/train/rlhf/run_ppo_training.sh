#!/bin/bash

# PPO Training Script
# Usage: ./run_ppo_training.sh

# Set variables
export MODEL_PATH="/mnt/e/CheckPoints/ChemicalFactory/output/output/qwen205_moltrans_mit_mixed_augm_rlhf_sft_lora_para1"
export PPO_DATA_PATH="/mnt/e/DataSets/Chemistry/ForwardPrediction/RLHF/mit_mixed/ppo/"
export PPO_DATA_FILE="MIT_mixed_augm.json"
export PPO_MODEL_OUTPUT_PATH="/mnt/e/CheckPoints/ChemicalFactory/output/output/qwen205_moltrans_mit_mixed_augm_rlhf_ppo_lora_para1"

# Training parameters
export BATCH_SIZE=4
export LEARNING_RATE=1.41e-5
export MAX_LENGTH=256
export NUM_EPOCHS=10
export SEED=42

# Run training
python ppo_training.py \
    --model_path "$MODEL_PATH" \
    --ppo_data_path "$PPO_DATA_PATH" \
    --ppo_data_file "$PPO_DATA_FILE" \
    --ppo_model_output_path "$PPO_MODEL_OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --num_epochs $NUM_EPOCHS \
    --seed $SEED

echo "Training script completed"