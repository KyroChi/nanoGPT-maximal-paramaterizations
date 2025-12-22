#!/bin/bash
#SBATCH --job-name=gpt2-moe-124m
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/moe_124m_%j.out
#SBATCH --error=logs/moe_124m_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Get config file path from command line argument, or use default
CONFIG_FILE="${1:-config/train_gpt2_moe_124m.py}"

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config file: $CONFIG_FILE"
echo "Start time: $(date)"

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))

# Launch training with torchrun
torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py "$CONFIG_FILE"

echo "End time: $(date)"

