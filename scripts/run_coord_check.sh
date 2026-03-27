#!/bin/bash
# Coordinate check: 1536w, sweep r (KV head count) for 20 iterations.
# Verifies muP scaling is correct across GQA ratios.
# Waits for any running experiment to finish first.
#
# Usage:
#   tmux new -s coordcheck
#   bash scripts/run_coord_check.sh
#   # Ctrl-B, D to detach
#
# Results go to coord_check_results/*.csv
# Pull to local: scp ubuntu@<ip>:~/nanoGPT-maximal-paramaterizations/coord_check_results/*.csv .

set -e
cd ~/nanoGPT-maximal-paramaterizations

LOG=/tmp/coord_check.log
exec > >(tee -a "$LOG") 2>&1

# Wait for exp1 to finish if it's running
if pgrep -f "run_exp1" > /dev/null; then
    echo "Experiment 1 is running, waiting for it to finish..."
    while pgrep -f "run_exp1" > /dev/null; do
        sleep 300
        echo "  still waiting... $(date)"
    done
    echo "Experiment 1 finished, starting coord check"
fi

echo "=== Coordinate check starting $(date) ==="

OUT_DIR=coord_check_results
mkdir -p $OUT_DIR

N_EMBD=1536
N_HEAD=12       # head_dim=128
N_LAYER=3
MAX_ITERS=20
IMPL=gqa_mup
MUP_MULT=$(python3 -c "print($N_EMBD / 256)")
SEEDS=(42 43 44)

# Sweep r: all valid divisors of n_head=12
# r=1 (kv=12), r=2 (kv=6), r=3 (kv=4), r=4 (kv=3), r=6 (kv=2), r=12 (kv=1)
KV_HEADS=(12 6 4 3 2 1)

for n_kv in "${KV_HEADS[@]}"; do
    r=$((N_HEAD / n_kv))
    for seed in "${SEEDS[@]}"; do
        tag="${N_EMBD}w_r${r}_kv${n_kv}_s${seed}"
        echo ""
        echo "=== r=${r} (kv=${n_kv}), seed=${seed} ==="
        uv run python gqa_mup/train.py \
            --n_embd=$N_EMBD --n_head=$N_HEAD --n_kv_head=$n_kv --n_layer=$N_LAYER \
            --batch_size=1 --gradient_accumulation_steps=1 \
            --max_iters=$MAX_ITERS --eval_interval=100000 --eval_iters=1 \
            --learning_rate=4e-5 --mup=True --mup_multiplier=$MUP_MULT \
            --impl=$IMPL --coord_check=True --wandb_log=False \
            --compile=False --dtype=float32 --out_dir=$OUT_DIR \
            --seed=$seed --tag=$tag
    done
done

echo ""
echo "=== Coordinate check done $(date) ==="
echo "Results in $OUT_DIR/"
ls -lh $OUT_DIR/*.csv
