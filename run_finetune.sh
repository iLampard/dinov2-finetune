#!/bin/bash

# =============================================================================
# Script to launch finetune.py with DeepSpeed for distributed training
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage instructions
usage() {
    echo "Usage: $0 --config <path_to_config_yaml> [--gpus <gpu_ids_comma_separated>]"
    echo ""
    echo "Options:"
    echo "  --config    Path to the finetune_config.yaml file"
    echo "  --gpus      (Optional) Comma-separated list of GPU IDs to use (e.g., 0,1,2,3)"
    echo ""
    echo "Example:"
    echo "  $0 --config ./finetune_config.yaml --gpus 0,1,2,3"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# Check if config_path is provided
if [ -z "$CONFIG_PATH" ]; then
    echo "Error: --config parameter is required."
    usage
fi

# If GPU_IDS is provided, update the config file with the specified GPU IDs
if [ ! -z "$GPU_IDS" ]; then
    echo "Updating GPU IDs in the config file to: $GPU_IDS"
    # Use yq to update the gpu_ids field in the distributed section
    # Ensure yq is installed: pip install yq
    yq -i ".distributed.gpu_ids = \"$GPU_IDS\"" "$CONFIG_PATH"
fi

# Extract the number of GPUs from the config file
NUM_GPUS=$(yq ".distributed.gpu_ids | split(\",\") | length" "$CONFIG_PATH")

# Validate NUM_GPUS
if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
    echo "Error: Unable to determine the number of GPUs from gpu_ids."
    exit 1
fi

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: At least one GPU must be specified for distributed training."
    exit 1
fi

echo "Launching finetune.py with DeepSpeed using $NUM_GPUS GPU(s)."

# Run finetune.py with DeepSpeed
deepspeed \
    --num_gpus=$NUM_GPUS \
    finetune.py \
    --config "$CONFIG_PATH"

echo "Training started successfully."