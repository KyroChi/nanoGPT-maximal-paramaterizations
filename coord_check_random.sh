#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --array=0-69
#SBATCH --mem=50G

# Stress test the coordinate check by pretty much randomizing everything!

# now=$(date +%Y-%m-%d_%H-%M-%S)
# use the unique slurm id as the directory name
now=${SLURM_ARRAY_JOB_ID}
# out_dir=coord-check-
out_dir=coord-check-impl/${now}
mkdir -p ${out_dir}

# Define parameters as arrays
# embeds=(256 288 384 480 512 640 768 896 1024 1280 1536 2048)
# embeds=(256 480 512 640 768 896 1024 1280 1536 2048)
# Constant head size
embeds=(256 320 384 448 512 640 768 896 960 1024 1088 1152 1760 2176)
head_size=(32 32 32 32 32 32 32 32 32 32 32 32 32 32)
kv_heads=(2 2 2 2 2 2 2 2 2 2 2 2 2 2)

embeds=(256 384 512 640 768 896 1024 1152 1280 1408 1536)
head_size=(32 48 64 80 96 112 128 144 160 176 192)
kv_heads=(2 2 2 2 2 2 2 2 2 2 2 2 2 2)

seeds=(42 43 44 45 46)

# Calculate indices based on SLURM array ID
emb_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))

# Get parameter values
emb=${embeds[$emb_index]}
head_size=${head_size[$emb_index]}
n_kv_heads=${kv_heads[$emb_index]}
seed=${seeds[$seed_index]}

n_heads=$(( emb / head_size ))
mup_multiplier=$(echo "scale=2; $emb / 256" | bc)

echo "mup_muliplier: ${mup_multiplier}, n_heads: ${n_heads}, emb: ${emb}, seed: ${seed}"
srun python coord_check.py \
    --out_dir=${out_dir} \
    --eval_interval=10000000 \
    --log_interval=10000000 \
    --eval_iters=1 \
    --eval_only=False \
    --init_from='scratch' \
    --wandb_log=False \
    --dataset='slim_pajama' \
    --gradient_accumulation_steps=1 \
    --batch_size=1 \
    --block_size=1024 \
    --n_layer=3 \
    --n_head=${n_heads} \
    --n_kv_head=${n_kv_heads} \
    --n_embd=${emb} \
    --dropout=0.0 \
    --bias=False \
    --init_std=0.02 \
    --learning_rate=4e-5 \
    --max_iters=4 \
    --weight_decay=0.0 \
    --beta1=0.9 \
    --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=False \
    --mup=True \
    --mup_multiplier=${mup_multiplier} \
    --seed=${seed} \
    --backend='nccl' \
    --device='cuda' \
    --dtype='float32' \
    --compile=False \
    --impl='tpv_left_impl'