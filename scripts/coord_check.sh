#!/bin/bash
# Coordinate checking for GQA-muP validation
# Usage (SLURM): sbatch --array=0-5 scripts/coord_check.sh
# Usage (local): bash scripts/coord_check.sh <index>
#
#SBATCH --time=5:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio

# Experiments: "head_size n_embd n_head n_kv_head"
EXPS=(
    "48 576 12 1"
    "48 576 12 2"
    "48 576 12 3"
    "48 576 12 4"
    "48 576 12 6"
    "48 576 12 12"
)

# Use SLURM array index if available, otherwise use first CLI arg
index=${SLURM_ARRAY_TASK_ID:-${1:-0}}

if [ $index -ge ${#EXPS[@]} ]; then
    echo "Index $index out of range (0-$((${#EXPS[@]}-1)))."
    exit 1
fi

item="${EXPS[$index]}"
head_size=$(echo $item | cut -d' ' -f1)
model_size=$(echo $item | cut -d' ' -f2)
n_heads=$(echo $item | cut -d' ' -f3)
n_kv_heads=$(echo $item | cut -d' ' -f4)

echo "head_size: ${head_size}, model_size: ${model_size}, n_heads: ${n_heads}, n_kv_heads: ${n_kv_heads}"

now=$(date +%Y-%m-%d_%H-%M-%S)
out_dir=coord-check-impl/${SLURM_ARRAY_JOB_ID:-local_${now}}
mkdir -p ${out_dir}

for seed in {0..10}; do
    mup_multiplier=$(echo "scale=2; $model_size / 256" | bc)

    python gqa_mup/train.py \
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
        --n_layer=8 \
        --n_head=${n_heads} \
        --n_kv_head=${n_kv_heads} \
        --n_embd=${model_size} \
        --dropout=0.0 \
        --bias=False \
        --init_std=0.02 \
        --learning_rate=4e-5 \
        --max_iters=4 \
        --eps=1e-10 \
        --weight_decay=0.0 \
        --beta1=0.9 \
        --beta2=0.95 \
        --grad_clip=1.0 \
        --decay_lr=False \
        --mup=True \
        --mup_multiplier=${mup_multiplier} \
        --seed=${seed} \
        --device='cuda' \
        --dtype='float32' \
        --compile=False \
        --coord_check=True \
        --impl='tpv_left_impl_no_kv' \
        --tag="${item}"
done
