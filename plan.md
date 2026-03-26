# Validation & Experiment Plan

## Overview

Two people, two machines, two experiments.

- **Kyle** validates the refactored code on a **1xH100 Lambda Labs box** by
  running a cheap width-transfer experiment end-to-end. This is the
  integration test. ~14 hours of compute, ~2 hours of active time.

- **H+M** run the expensive experiments on their **SLURM cluster** (unlimited
  GPUs). They clone, install, and run — zero debugging. Kyle's validation
  proves the code works; H+M just scale it up.

---

## Kyle's Validation (1xH100)

### Goal

Prove the refactored code works by running a small width-transfer
experiment (Experiment 1) and inspecting the results. If the muP curve
shows LR transfer and the SP curve doesn't, the code is correct.

### Step 1: GitHub auth + fresh setup

Lambda Labs instances don't have your GitHub credentials. Set up SSH key
auth so you can clone private repos and push:

```bash
# Generate a new SSH key (hit enter for all prompts, no passphrase needed)
ssh-keygen -t ed25519 -C "kyle-lambda" -f ~/.ssh/id_ed25519 -N ""

# Print the public key
cat ~/.ssh/id_ed25519.pub
```

Copy the public key output, then add it to GitHub:
1. Go to https://github.com/settings/keys
2. Click "New SSH key"
3. Title: "Lambda Labs H100" (or whatever)
4. Paste the key
5. Click "Add SSH key"

Test it works:
```bash
ssh -T git@github.com
# Should say: "Hi KyroChi! You've successfully authenticated..."
```

Now clone:
```bash
git clone git@github.com:KyroChi/nanoGPT-maximal-paramaterizations.git
cd nanoGPT-maximal-paramaterizations
git checkout paper-release

# Set git identity for commits
git config user.email "your@email.com"
git config user.name "Kyle Chickering"

# Install
uv sync --group data --group analysis
```

Verify GPU:
```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Step 2: Prepare OpenWebText

```bash
uv run python data/prepare_openwebtext.py
```

Takes ~1-2 hours. Produces `data/train.bin` (~17GB) and `data/val.bin` (~8.5MB).

### Step 3: Benchmark

```bash
uv run python scripts/benchmark.py --output benchmark_results.json
```

This gives real sec/iter and MFU numbers. Note the recommended batch_size.

### Step 4: Smoke test (5 minutes)

Run a tiny training job at each scale to verify nothing crashes:

```bash
# Proxy (256w, 28M params)
uv run python gqa_mup/train.py \
    --n_embd=256 --n_head=4 --n_kv_head=2 --n_layer=3 \
    --batch_size=64 --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=1.0 \
    --impl=gqa_mup --wandb_log=False --compile=False --dtype=bfloat16

# Target (1024w, 137M params)
uv run python gqa_mup/train.py \
    --n_embd=1024 --n_head=16 --n_kv_head=4 --n_layer=3 \
    --batch_size=32 --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=4.0 \
    --impl=gqa_mup --wandb_log=False --compile=False --dtype=bfloat16
```

Both should print decreasing loss values. No crashes = good.

### Step 5: Coord check (10 minutes)

Quick sanity check that muP scaling is applied correctly:

```bash
for width in 256 512 1024; do
    n_head=$((width / 64))
    mup_mult=$(echo "$width / 256" | bc -l)
    uv run python gqa_mup/train.py \
        --n_embd=$width --n_head=$n_head --n_kv_head=2 --n_layer=3 \
        --batch_size=1 --max_iters=4 --eval_interval=10000 --eval_iters=1 \
        --learning_rate=4e-5 --mup=True --mup_multiplier=$mup_mult \
        --impl=gqa_mup --coord_check=True --wandb_log=False \
        --compile=False --dtype=float32 --out_dir=coord_check_test \
        --seed=42
done
```

Verify CSVs appear in `coord_check_test/`. Activation norms should be
roughly constant across widths (that's the whole point of muP).

### Step 6: Run Experiment 1 — Width Transfer (~14 hours)

This is the real test. Set up W&B first:

```bash
export WANDB_API_KEY=<your-key>
```

Then run:

```bash
python experiments/local_orchestrator.py \
    --config_generator_file experiments/configs/width_only.py
```

This runs the width_only config which sweeps LR at 256w and 1024w widths,
comparing SP vs muP. With the existing config settings (11 LRs, 2 seeds,
5 model widths) it may be more runs than needed. To run a quick subset:

```bash
python experiments/local_orchestrator.py \
    --config_generator_file experiments/configs/width_only.py \
    --max-experiments 20
```

Alternatively, create a minimal width transfer config with just 2 scales,
7 LRs, 3 seeds, and 4 TPP.

**What to look for in W&B:**
- Loss-vs-LR curves should be smooth parabolas (not noisy messes)
- For muP: the optimal LR should be at roughly the same place for both
  widths. The curves should overlap.
- For SP: the optimal LR should shift rightward as width increases.
- If you see this pattern, the code is working correctly.

### Step 7: Fix and push

```bash
# Fix any issues found in steps 4-6
git add -A
git commit -m "validated on H100"
git push
```

### Estimated timeline

| Step | Active time | Wall clock |
|------|------------|------------|
| Setup + data prep | 10 min | 1-2 hours |
| Benchmark | 5 min | 10 min |
| Smoke tests | 10 min | 5 min |
| Coord check | 5 min | 10 min |
| Experiment 1 | 0 (runs overnight) | ~14 hours |
| Inspect W&B + fix | 30 min | 30 min |
| **Total** | **~1 hour** | **~16 hours** |

---

## H+M's Cluster Runs

### What they run

**Experiment 2: GQA ablation** — Varies the KV repetition factor r at
fixed width 1536, comparing `mup_no_kv` vs `gqa_mup`. Shows that the
GQA-specific scaling is necessary.

- Config: `experiments/configs/r_ablations_2.py`
- 7 r-values × 11 LRs × 3 seeds × 2 impls = 462 runs
- Each run: ~1 hour on 1xH100 at 4 TPP
- On cluster with parallel jobs: minutes of wall clock

**Experiment 3: Large-scale transfer** — 3-scale width transfer
(256w→1024w→4096w) comparing SP vs GQA-muP. Directly addresses reviewer
concerns about scale separation.

- Config: `experiments/configs/reviewer_transfer.py`
- 96 runs (3 scales × 2 params × {9,7,7} LRs × {3,2,1} seeds)
- Largest model: 945M params at 4096w
- On cluster with parallel jobs: a few hours wall clock

### Setup instructions for H+M

See `HANDOFF.md` for the complete step-by-step.

### What H+M should NOT do

- Do not modify any files in `gqa_mup/` or the experiment configs
- Do not change impl names, LR ranges, or seed lists
- Do not debug Python errors — report them to Kyle
- Hardware-specific settings (batch_size, n_gpus, partition) can be
  overridden via `--hardware` flag without touching the configs

---

## Compute Budget Summary

All estimates at 40% MFU on H100 (989 TFLOPS BF16 peak). Real MFU
will be measured by benchmark.py.

### Experiment 1: Width transfer (Kyle, 1xH100)

| Scale | Width | Params | 1 run (4 TPP) | Full sweep |
|-------|------:|-------:|--------------:|-----------:|
| Proxy | 256 | 28M | 0.01h | — |
| Target | 1024 | 137M | 0.32h | — |
| **Total** | | | | **~14h** |

(7 LR × 3 seeds × 2 params × 2 scales = 84 runs)

### Experiment 2: GQA ablation (H+M, cluster)

| r | kv_heads | Params | 1 run (4 TPP) |
|--:|---------:|-------:|--------------:|
| 1 | 24 | 241M | 0.98h |
| 2 | 12 | 234M | 0.92h |
| 3 | 8 | 232M | 0.90h |
| 4 | 6 | 230M | 0.89h |
| 6 | 4 | 229M | 0.89h |
| 8 | 3 | 229M | 0.88h |
| 12 | 2 | 228M | 0.88h |

Total: **266h on 1xH100** = ~33h on 8xH100 = minutes on cluster.

### Experiment 3: Large-scale transfer (H+M, cluster)

| Scale | Width | Params | 1 run |
|-------|------:|-------:|------:|
| Proxy | 256 | 28M | seconds |
| Mid | 1024 | 137M | 0.3h |
| Target | 4096 | 945M | 1-11h (varies by TPP) |

Total: depends on TPP choice. At 3 TPP: ~23h on 1xH100 = trivial on cluster.
