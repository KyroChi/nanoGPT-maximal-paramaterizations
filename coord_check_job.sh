#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

now=$(date +%Y-%m-%d_%H-%M-%S)
out_dir=coord-check-impl/${now}
mkdir -p ${out_dir}

head_size=64

embeds=(256 512 1024 2048)
for seed in {0..5}; do
    for emb in "${embeds[@]}"; do
        mup_multiplier=$(( emb / 256 ))
        n_heads=$(( emb / head_size ))

        echo "mup_muliplier: ${mup_multiplier}, n_heads: ${n_heads}, emb: ${emb}, seed: ${seed}"
        srun python coord_check.py \
            --out_dir=${out_dir} \
            --eval_interval=10000000 \
            --log_interval=10000000 \
            --eval_iters=1 \
            --eval_only=False \
            --init_from='scratch' \
            --wandb_log=False \
            --dataset='shakespeare_char' \
            --gradient_accumulation_steps=1 \
            --batch_size=32 \
            --block_size=1024 \
            --n_layer=3 \
            --n_head=${n_heads} \
            --n_embd=${emb} \
            --dropout=0.0 \
            --bias=False \
            --init_std=0.02 \
            --learning_rate=1e-2 \
            --max_iters=12 \
            --weight_decay=1.0 \
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
            --compile=False
    done
done