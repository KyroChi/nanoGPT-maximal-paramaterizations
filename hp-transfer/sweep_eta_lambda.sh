#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

now=$(date +%Y-%m-%d_%H-%M-%S)
out_dir=mu-transfer-char/${now}
mkdir -p $out_dir

head_size=64

embeds=(256 512 1024 2048 4096)
etas=()
lambdas=()

srun python hp_train.py \
    --out_dir=${out_dir} \
    --wandb_log=True \
    --wandb_project='mu-transfer-char' \
    --wandb_run_name="training_shakedown" \
    --eval_interval=20 \
    --log_interval=1 \
    --eval_iters=100 \
    --eval_only=False \
    --init_from='scratch' \
    --dataset='shakespeare_char' \
    --gradient_accumulation_steps=1 \
    --batch_size=64 \
    --block_size=1024 \
    --n_layer=3 \
    --n_head=4 \
    --n_embd=256 \
    --dropout=0.0 \
    --bias=False \
    --init_std=0.02 \
    --learning_rate=2e-4 \
    --max_iters=500 \
    --weight_decay=1e-1 \
    --beta1=0.9 \
    --beta2=0.95 \
    --grad_clip=1.0 \
    --decay_lr=False \
    --mup=True \
    --mup_multiplier=1 \
    --seed=42 \
    --backend='nccl' \
    --device='cuda' \
    --dtype='float32' \
    --compile=False \
    --coord_check=False

# for seed in {0..5}; do
#     for emb in "${embeds[@]}"; do
#         for eta in "${etas[@]}"; do
#             for lambda in "${lambdas[@]}"; do
#                 mup_multiplier=$(( emb / 256 ))
#                 n_heads=$(( emb / head_size ))

#                 echo "mup_multiplier: ${mup_multiplier}, n_heads: ${n_heads}, emb: ${emb}, seed: ${seed}, eta: ${eta}, lambda: ${lambda}"
#                 srun python train.py \
#                     --out_dir=${out_dir} \
#                     --wandb_log=True \
#                     --wandb_project='mu-transfer-char' \
#                     --wandb_run_name="mup_${mup_multiplier}_heads_${n_heads}_emb_${emb}_seed_${seed}_eta_${eta}_lambda_${lambda}" \
#                     --eval_interval=10000000 \
#                     --log_interval=10000000 \
#                     --eval_iters=1 \
#                     --eval_only=False \
#                     --init_from='scratch' \
#                     --wandb_log=False \
#                     --dataset='shakespeare_char' \
#                     --gradient_accumulation_steps=1 \
#                     --batch_size=1 \
#                     --block_size=1024 \
#                     --n_layer=3 \
#                     --n_head=${n_heads} \
#                     --n_embd=${emb} \
#                     --dropout=0.0 \
#                     --bias=False \
#                     --init_std=0.02 \
#                     --learning_rate=${eta} \
#                     --max_iters=12 \
#                     --weight_decay=${lambda} \
#                     --beta1=0.9 \
#                     --beta2=0.95 \
#                     --grad_clip=1.0 \
#                     --decay_lr=False \
#                     --mup=True \
#                     --mup_multiplier=${mup_multiplier} \
#                     --seed=${seed} \
#                     --eta=${eta} \
#                     --lambda=${lambda} \
#                     --backend='nccl' \
#                     --device='cuda' \
#                     --dtype='float32' \
#                     --compile=False \
#                     --coord_check=False
#             done
#         done
#     done
# done

