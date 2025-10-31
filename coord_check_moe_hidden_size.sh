#!/bin/bash
# filepath: /mnt/weka/home/kyle.chickering/code/nanoGPT/coord_check_moe.sh
#SBATCH --time=1:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-39
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --distribution=pack

# But then we will use topk vals, as well as vary number of experts.
# Set fixed values as requested
TOPK=("8")
EXPS=("64")

# Vary ffn hidden sizes from 384 to 4*384 (i.e., 384, 768, 1152, 1536)
FFN_SIZES=(384 768 1152 1536)

# Calculate the index into the FFN_SIZES array
index=$SLURM_ARRAY_TASK_ID

# Calculate the indices
ffn_idx=$((index % ${#FFN_SIZES[@]}))
# Use fixed n_experts and topk
n_experts=${EXPS[0]}
topk=${TOPK[0]}
ffn_hidden_size=${FFN_SIZES[$ffn_idx]}
moe_ffn_mup_multiplier=$(echo "scale=4; ${ffn_hidden_size} / $((4 * 384))" | bc)

    now=$(date +%Y-%m-%d_%H-%M-%S)
    out_dir=coord-check-impl/${SLURM_ARRAY_JOB_ID}
    mkdir -p ${out_dir}

    for seed in {0..5}; 
    do
        srun python coord_check.py \
            --out_dir=${out_dir} \
            --eval_interval=10000000 \
            --log_interval=10000000 \
            --eval_iters=1 \
            --eval_only=False \
            --init_from='scratch' \
            --wandb_log=False \
            --dataset='openwebtext' \
            --gradient_accumulation_steps=1 \
            --batch_size=1 \
            --block_size=1024 \
            --n_layer=4 \
            --n_head=16 \
            --n_kv_head=16 \
            --n_embd=384 \
            --dropout=0.0 \
            --bias=False \
            --init_std=0.02 \
            --learning_rate=4e-5 \
            --max_iters=4 \
            --weight_decay=0.0 \
            --beta1=0.9 \
            --beta2=0.95 \
            --eps=1e-16 \
            --grad_clip=0.0 \
            --decay_lr=False \
            --mup=True \
            --seed=${seed} \
            --backend='nccl' \
            --device='cuda' \
            --dtype='float32' \
            --compile=False \
            --impl='moe_base' \
            --tag="${n_experts}_${ffn_hidden_size}" \
            --use_moe=True \
            --num_experts=${n_experts} \
            --moe_ffn_hidden_size=${ffn_hidden_size} \
            --router_topk=${topk} \
            --moe_ffn_mup_multiplier=${moe_ffn_mup_multiplier} \
            --moe_random_router=False
    done
else
    echo "SLURM_ARRAY_TASK_ID is out of range (0-35)."
fi