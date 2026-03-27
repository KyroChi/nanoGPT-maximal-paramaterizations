#!/bin/bash
# Run coordinate checks across widths to verify muP scaling.
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

WIDTHS=(256 512 1024 1536)
HEAD_DIM=128
N_LAYER=3
SEEDS=(42 43 44)
IMPL=gqa_mup

for width in "${WIDTHS[@]}"; do
    n_head=$((width / HEAD_DIM))
    # Need at least 1 head
    if [ $n_head -lt 1 ]; then
        n_head=1
    fi
    n_kv=$((n_head > 1 ? n_head / 2 : 1))
    mup_mult=$(python3 -c "print($width / 256)")

    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "=== width=${width}, n_head=${n_head}, n_kv=${n_kv}, seed=${seed} ==="
        uv run python gqa_mup/train.py \
            --n_embd=$width --n_head=$n_head --n_kv_head=$n_kv --n_layer=$N_LAYER \
            --batch_size=1 --gradient_accumulation_steps=1 \
            --max_iters=4 --eval_interval=100000 --eval_iters=1 \
            --learning_rate=4e-5 --mup=True --mup_multiplier=$mup_mult \
            --impl=$IMPL --coord_check=True --wandb_log=False \
            --compile=False --dtype=float32 --out_dir=$OUT_DIR \
            --seed=$seed
    done
done

echo ""
echo "=== Coordinate check done $(date) ==="
echo "Results in $OUT_DIR/"
ls -lh $OUT_DIR/*.csv
