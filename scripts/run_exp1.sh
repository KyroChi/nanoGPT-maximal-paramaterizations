#!/bin/bash
# GQA ablation experiment: SP vs gqa_mup, r=1 vs r=12
# 1536w, 3L, head_dim=128, batch=128 (64 micro × 2 accum), 5 TPP
#
# Order: gqa_mup r=1 (all LRs) -> gqa_mup r=12 -> SP r=1 -> SP r=12
#
# Usage:
#   tmux new -s exp1
#   bash scripts/run_exp1.sh
#   # Ctrl-B, D to detach

set -e
cd ~/nanoGPT-maximal-paramaterizations

LOG=/tmp/exp1.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Experiment 1 starting $(date) ==="

WANDB_PROJECT="gqa-mup-exp1-ablation"
N_EMBD=1536
N_HEAD=12       # head_dim = 1536/12 = 128
N_LAYER=3
BLOCK_SIZE=1024
BATCH_SIZE=64
GRAD_ACCUM=2    # effective batch = 128
SEED=42
TPP=5
DTYPE=bfloat16

# 5 LRs: sqrt(2)-spaced around 5e-3
LRS=(0.002500 0.003536 0.005000 0.007071 0.010000)

# Run order: gqa_mup first, then SP
# Within each: r=1 first, then r=12
CONFIGS=(
    "gqa_mup True  12 r1  9193"
    "gqa_mup True   1 r12 8697"
    "sp      False 12 r1  9193"
    "sp      False  1 r12 8697"
)

run_idx=0
total_runs=20  # 4 configs × 5 LRs

for config_line in "${CONFIGS[@]}"; do
    read -r impl mup n_kv kv_label max_iters <<< "$config_line"

    if [ "$mup" = "True" ]; then
        mup_mult=$(echo "scale=4; $N_EMBD / 256" | bc)
    else
        mup_mult=1
    fi

    for lr in "${LRS[@]}"; do
        run_idx=$((run_idx + 1))
        min_lr=$(python3 -c "print($lr / 10)")
        run_name="${impl}_${kv_label}_lr${lr}_s${SEED}"

        echo ""
        echo "=== Run $run_idx/$total_runs: $run_name (${max_iters} iters) ==="
        echo "    impl=$impl mup=$mup kv=$n_kv lr=$lr $(date)"

        uv run python gqa_mup/train.py \
            --n_embd=$N_EMBD --n_head=$N_HEAD --n_kv_head=$n_kv --n_layer=$N_LAYER \
            --batch_size=$BATCH_SIZE --gradient_accumulation_steps=$GRAD_ACCUM \
            --block_size=$BLOCK_SIZE --max_iters=$max_iters \
            --eval_interval=500 --eval_iters=10 \
            --learning_rate=$lr --min_lr=$min_lr \
            --weight_decay=0.1 --warmup_iters=200 \
            --decay_profile=cosine --decay_lr=True \
            --mup=$mup --mup_multiplier=$mup_mult \
            --impl=$impl --seed=$SEED \
            --wandb_log=True --wandb_project=$WANDB_PROJECT \
            --wandb_run_name=$run_name \
            --compile=False --dtype=$DTYPE
    done
done

echo ""
echo "=== ALL DONE $(date) ==="
echo "Check W&B: project $WANDB_PROJECT"
echo "Check logs: cat /tmp/exp1.log"
