#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

srun python reverse_lambda_sweep.py \
    --d_emb=$1 \
    --seed=$2 \
    --base_dim=$3 \
    --lr_samples=$4 \
    --tpp=$5 \
    --experiment_name=$6 \
    --batch_size=$7 \
    --gradient_accumulation_steps=$8