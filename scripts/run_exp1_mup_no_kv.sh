#!/bin/bash
# Experiment 1b: muP WITHOUT GQA correction, r=1 vs r=12
# Same setup as run_exp1.sh but with mup_no_kv impl only.
# This is the ablation baseline — shows what happens if you use
# standard muP scaling on KV layers without the GQA correction.
#
# Usage:
#   bash scripts/run_exp1_mup_no_kv.sh

set -e
cd "$(dirname "$0")/.."

LOG=/tmp/exp1_mup_no_kv.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Experiment 1b (mup_no_kv) starting $(date) ==="

WANDB_PROJECT="gqa-mup-exp1-ablation"
N_EMBD=1536
N_HEAD=12
N_LAYER=3
BLOCK_SIZE=1024
BATCH_SIZE=32
GRAD_ACCUM=2
SEED=42
TPP=5
DTYPE=bfloat16

LRS=(0.000312 0.000442 0.000625 0.000884 0.001250 0.001768 0.002500 0.003536 0.005000)

CONFIGS=(
    "mup_no_kv True 12 r1  18386"
    "mup_no_kv True  1 r12 17395"
)

run_idx=0
total_runs=18

for config_line in "${CONFIGS[@]}"; do
    read -r impl mup n_kv kv_label max_iters <<< "$config_line"
    mup_mult=$(python3 -c "print($N_EMBD / 256)")

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
echo "Check logs: cat /tmp/exp1_mup_no_kv.log"
