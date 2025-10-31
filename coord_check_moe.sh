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
EXPS=("16" "32" "64" "96" "128" "160" "192")
# TOPK=("2" "4" "8" "16" "32")
TOPK=("2" "4" "6" "8")

# Calculate the index into the EXPS and FFN_DIM arrays
index=$SLURM_ARRAY_TASK_ID

# Calculate the row and column indices for the grid
row=$((index / ${#TOPK[@]}))
col=$((index % ${#TOPK[@]}))

n_embd=768
ffn_base_hidden_size=$((4 * n_embd)) 


# Check if the index is within the bounds of the array
if [ $row -ge 0 ] && [ $row -lt ${#EXPS[@]} ] && [ $col -ge 0 ] && [ $col -lt ${#TOPK[@]} ]; then
    n_experts=${EXPS[$row]}
    # n_experts=${TOPK[$col]}
    topk=${TOPK[$col]}
    ffn_hidden_size=$((ffn_base_hidden_size / topk))
    moe_ffn_mup_multiplier=$(echo "scale=2; ${ffn_hidden_size} / ${ffn_base_hidden_size}" | bc)

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
            --n_embd=${n_embd} \
            --dropout=0.0 \
            --bias=False \
            --init_std=0.02 \
            --learning_rate=4e-5 \
            --max_iters=4 \
            --weight_decay=0.0 \
            --beta1=0.9 \
            --beta2=0.95 \
            --eps=1e-6 \
            --grad_clip=0.0 \
            --decay_lr=False \
            --mup=True \
            --seed=${seed} \
            --backend='nccl' \
            --device='cuda' \
            --dtype='float32' \
            --compile=False \
            --impl='moe_base' \
            --tag="${n_experts}_${topk}" \
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