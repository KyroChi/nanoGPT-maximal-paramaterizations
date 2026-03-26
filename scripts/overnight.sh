#!/bin/bash
# Launch in tmux then go to bed:
#   tmux new -s overnight
#   bash scripts/overnight.sh
#   # Ctrl-B, D to detach
#
# In the morning:
#   tmux attach -t overnight
#   # or check: cat /tmp/overnight.log

set -e
cd ~/nanoGPT-maximal-paramaterizations

LOG=/tmp/overnight.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Starting $(date) ==="

# Step 1: Prepare data (the long part)
if [ -f data/train.bin ] && [ -f data/val.bin ]; then
    echo "Data already exists, skipping prep"
else
    echo "=== Preparing OpenWebText (this takes 1-2 hours) ==="
    uv run python data/prepare_openwebtext.py
fi

echo "=== Data ready $(date) ==="
ls -lh data/train.bin data/val.bin

# Step 2: Smoke test 256w
echo ""
echo "=== Smoke test: 256w proxy ==="
uv run python gqa_mup/train.py \
    --n_embd=256 --n_head=4 --n_kv_head=2 --n_layer=3 \
    --batch_size=64 --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=1.0 \
    --impl=gqa_mup --wandb_log=False --compile=False --dtype=bfloat16

# Step 3: Smoke test 1024w
echo ""
echo "=== Smoke test: 1024w target ==="
uv run python gqa_mup/train.py \
    --n_embd=1024 --n_head=16 --n_kv_head=4 --n_layer=3 \
    --batch_size=32 --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=4.0 \
    --impl=gqa_mup --wandb_log=False --compile=False --dtype=bfloat16

# Step 4: Benchmark
echo ""
echo "=== Benchmark ==="
uv run python scripts/benchmark.py --output benchmark_results.json

echo ""
echo "=== ALL DONE $(date) ==="
echo "Check results: cat /tmp/overnight.log"
