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

# Step 2: Smoke test 256w (with wandb)
echo ""
echo "=== Smoke test: 256w proxy ==="
uv run python gqa_mup/train.py \
    --n_embd=256 --n_head=2 --n_kv_head=1 --n_layer=3 \
    --batch_size=32 --gradient_accumulation_steps=1 \
    --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=1.0 \
    --impl=gqa_mup --wandb_log=True --wandb_project=gqa-mup-smoke \
    --wandb_run_name=smoke_256w --compile=False --dtype=bfloat16

# Step 3: Smoke test 1536w (with wandb)
echo ""
echo "=== Smoke test: 1536w ==="
uv run python gqa_mup/train.py \
    --n_embd=1536 --n_head=12 --n_kv_head=1 --n_layer=3 \
    --batch_size=8 --gradient_accumulation_steps=1 \
    --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=6.0 \
    --impl=gqa_mup --wandb_log=True --wandb_project=gqa-mup-smoke \
    --wandb_run_name=smoke_1536w --compile=False --dtype=bfloat16

echo ""
echo "=== Smoke tests done $(date), starting Experiment 1 ==="

# ================================================================
# Experiment 1: GQA ablation at 1536w, depth=3, head_dim=128
# r=1 (kv=12) vs r=12 (kv=1)
# 5 LRs (sqrt(2)-spaced around 5e-3) × 1 seed × 2 params (SP + gqa_mup)
# 5 TPP per run
# Estimated: ~84 hours on 1xA100
# ================================================================

WANDB_PROJECT="gqa-mup-exp1-ablation"
N_EMBD=1536
N_HEAD=12
N_LAYER=3
BLOCK_SIZE=1024
BATCH_SIZE=16
SEED=42
TPP=5
DTYPE=bfloat16

# 5 LRs: sqrt(2)-spaced around 5e-3
LRS=(0.002500 0.003536 0.005000 0.007071 0.010000)

# r=1 (kv=12) and r=12 (kv=1)
KV_HEADS=(12 1)
KV_LABELS=(r1 r12)

# SP and GQA-muP
IMPLS=(sp gqa_mup)
MUP_FLAGS=(False True)

run_idx=0
total_runs=20  # 5 LR × 2 kv × 2 impl

for impl_idx in 0 1; do
    impl=${IMPLS[$impl_idx]}
    mup=${MUP_FLAGS[$impl_idx]}

    if [ "$mup" = "True" ]; then
        mup_mult=$(echo "scale=4; $N_EMBD / 256" | bc)
    else
        mup_mult=1
    fi

    for kv_idx in 0 1; do
        n_kv=${KV_HEADS[$kv_idx]}
        kv_label=${KV_LABELS[$kv_idx]}

        # Compute max_iters for 5 TPP
        # params ≈ 230M, tokens = params * 5 ≈ 1.15B
        # tokens_per_iter = batch_size * block_size = 16 * 1024 = 16384
        # max_iters = 1.15B / 16384 ≈ 70190
        # Be conservative, compute from actual param count
        params=$(python3 -c "
ne=$N_EMBD; nl=$N_LAYER; nh=$N_HEAD; nkv=$n_kv; hd=ne//nh
pl = 2*ne**2 + 2*ne*(nkv*hd) + 2*ne*4*ne
non = 2*50304*ne + 1024*ne
p = non + nl*pl
iters = int(p * $TPP / ($BATCH_SIZE * $BLOCK_SIZE))
print(iters)
")

        for lr in "${LRS[@]}"; do
            run_idx=$((run_idx + 1))
            min_lr=$(python3 -c "print($lr / 10)")
            run_name="${impl}_${kv_label}_lr${lr}_s${SEED}"

            echo ""
            echo "=== Run $run_idx/$total_runs: $run_name (max_iters=$params) ==="
            echo "    impl=$impl mup=$mup kv=$n_kv lr=$lr $(date)"

            uv run python gqa_mup/train.py \
                --n_embd=$N_EMBD --n_head=$N_HEAD --n_kv_head=$n_kv --n_layer=$N_LAYER \
                --batch_size=$BATCH_SIZE --max_iters=$params --block_size=$BLOCK_SIZE \
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
done

echo ""
echo "=== ALL DONE $(date) ==="
echo "Check W&B: project $WANDB_PROJECT"
echo "Check logs: cat /tmp/overnight.log"
