# A100 Validation & Mu-Transfer Experiment Plan

This file is a plan for Claude to execute on a 1xA100 box. Read CLAUDE.md first
for setup instructions, then follow the phases below in order.

## Context

This is the codebase for the GQA-muP paper (https://openreview.net/forum?id=UJB2uOS9MR).
We need to: verify the cleaned-up code works, polish notebooks for collaborators,
and run a small-scale mu-transfer experiment showing that our GQA scaling rules
outperform standard parameterization.

Previous mu-transfer attempts were very noisy because we used thin networks
(small width like 256-384). Thin networks have high variance in loss curves,
which makes it hard to see the transfer signal. The fix is to use wider proxy
models (wider base width) and more training tokens.

---

## Phase 1: Verify Everything Works

### 1.1 Environment setup

```bash
uv sync --group data --group analysis
```

If `uv` is not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Verify torch sees the GPU:
```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 1.2 Prepare OpenWebText

```bash
uv run python data/prepare_openwebtext.py
```

This takes a while (~1-2 hours). It downloads ~54GB and produces:
- `data/train.bin` (~17GB, ~9B tokens)
- `data/val.bin` (~8.5MB, ~4M tokens)

Verify the files exist and have reasonable sizes.

### 1.3 Smoke test: single training run

Run a tiny training job to verify the pipeline works end-to-end:

```bash
uv run python gqa_mup/train.py \
    --n_embd=384 --n_head=6 --n_kv_head=2 --n_layer=4 \
    --batch_size=32 --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=1.5 \
    --impl='tpv_left_impl_new_kv_2' \
    --wandb_log=False --compile=False --device=cuda --dtype=bfloat16
```

Expected: it should print loss values that decrease over 50 iterations, no crashes.

Then test with compile=True:
```bash
uv run python gqa_mup/train.py \
    --n_embd=384 --n_head=6 --n_kv_head=2 --n_layer=4 \
    --batch_size=32 --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=1.5 \
    --impl='tpv_left_impl_new_kv_2' \
    --wandb_log=False --compile=True --device=cuda --dtype=bfloat16
```

### 1.4 Smoke test: standard parameterization

```bash
uv run python gqa_mup/train.py \
    --n_embd=384 --n_head=6 --n_kv_head=2 --n_layer=4 \
    --batch_size=32 --max_iters=50 --eval_interval=25 --eval_iters=5 \
    --learning_rate=3e-4 --mup=False --mup_multiplier=1 \
    --impl='standard_param_impl' \
    --wandb_log=False --compile=False --device=cuda --dtype=bfloat16
```

### 1.5 Smoke test: coordinate checking

```bash
uv run python gqa_mup/train.py \
    --n_embd=384 --n_head=6 --n_kv_head=2 --n_layer=4 \
    --batch_size=1 --max_iters=4 --eval_interval=10000 --eval_iters=1 \
    --learning_rate=4e-5 --mup=True --mup_multiplier=1.5 \
    --impl='tpv_left_impl_new_kv_2' \
    --coord_check=True --wandb_log=False --compile=False \
    --device=cuda --dtype=float32 --out_dir=coord_check_test
```

Verify it produces a CSV file in `coord_check_test/`.

### 1.6 Smoke test: local orchestrator

```bash
python experiments/local_orchestrator.py \
    --config_generator_file experiments/configs/width_only.py \
    --dry-run
```

Verify it prints experiment commands without errors.

### 1.7 Fix any issues

If any of the above fail, debug and fix. Common issues:
- Import paths may need adjusting (gqa_mup.* imports)
- The `configurator.py` exec pattern may need the working directory to be repo root
- CUDA memory: if OOM, reduce batch_size

Commit any fixes before proceeding.

---

## Phase 2: Polish Notebooks

The notebooks in `analysis/notebooks/` need to be cleaned up so collaborators
can understand them. For each notebook:

### General instructions for ALL notebooks:
1. Read the notebook fully
2. Add a markdown cell at the top with: title, one-paragraph description of
   what this notebook does and what figures it produces, and any data dependencies
3. Clear all outputs, then re-run from top to bottom to verify it works
4. Add brief markdown section headers between logical sections
5. Remove any dead/commented-out code cells
6. If a cell produces a figure, add a markdown cell above it explaining
   what the figure shows and why it matters for the paper

### Notebook-specific notes:

**gqa_math.ipynb** — GQA spectral norm analysis. This is important for the
paper — it validates the theoretical spectral norm predictions. Make sure
the key takeaway (spectral norms are stable across GQA configs) is clear.

**paper_expected_operator.ipynb** — Operator norm bounds. Exports a PDF
figure. Make sure the export path is relative (not absolute).

**scaling_laws.ipynb** — Chinchilla scaling law reproduction. This is
reference material, not core to our paper. Light polish only.

**slimpj_math.ipynb** — Tensor norm scaling verification. Brief notebook.
Just add a title and one-line description.

**transformer_sizing.ipynb** — Model sizing calculations. Useful reference.
Add a title, verify the math, light polish.

**weight_decay_annealing_hypothesis.ipynb** — WD annealing motivation.
This may or may not be in the final paper. Polish and add explanations
for the toy optimization examples.

Commit after polishing all notebooks.

---

## Phase 3: Mu-Transfer Experiment

### 3.1 Goal

Show that the GQA-muP scaling rules transfer hyperparameters (specifically
learning rate) across model widths better than standard parameterization.

The experiment:
1. Train models at a **proxy width** (small) with a sweep of learning rates
2. Find the optimal LR at proxy width
3. Train models at a **target width** (larger) with the SAME learning rate
4. Show that GQA-muP's optimal LR at proxy width is also near-optimal at
   target width, while SP's optimal LR does NOT transfer

### 3.2 Why previous attempts were noisy

Previous experiments used thin networks (n_embd=256 or 384 with 4-8 layers).
At these sizes:
- Loss variance between random seeds is high
- The LR-vs-loss curve is flat near the optimum, making the optimum hard to locate
- Small batch sizes compound the noise

### 3.3 Design for less noise

Key changes from previous attempts:
- **Wider base width**: Use n_embd=768 as proxy, n_embd=1536 as target
  (mup_multiplier = 768/256 = 3.0 for proxy, 1536/256 = 6.0 for target)
- **More depth**: n_layer=12 (deeper networks are more stable)
- **Larger batch**: batch_size=64 with gradient_accumulation_steps=4
  (effective batch = 256 sequences = 262K tokens/step)
- **More iterations**: 2000 steps (~500M tokens) — enough to see real convergence
- **3 seeds each**: for error bars
- **bfloat16**: for speed on A100
- **Cosine decay**: with warmup_iters=200

### 3.4 Create the experiment config

Create a NEW config file `experiments/configs/mu_transfer_1gpu.py` with this design:

```python
"""
Mu-transfer experiment: SP vs GQA-muP on 1xA100.
Shows that GQA-muP transfers optimal LR from proxy (768) to target (1536).

Usage:
    python experiments/local_orchestrator.py \
        --config_generator_file experiments/configs/mu_transfer_1gpu.py
"""
import json
import numpy as np
from copy import deepcopy

WANDB_PROJECT = 'mu-transfer-1gpu'

# --- Shared config ---
shared = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 100,
    'eval_iters': 20,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': 0.1,
    'warmup_iters': 200,
    'dataset': 'openwebtext',
    'block_size': 1024,
    'compile': 'true',
    'dtype': 'bfloat16',
    'n_gpus': 1,
    'bias': 'false',
    'dropout': 0.0,
}

# --- Model sizes ---
# Proxy: 768 wide, Target: 1536 wide
# Both use GQA: n_kv_head < n_head
# Fixed head_dim=64, so n_head = n_embd/64
models = [
    {   # Proxy model (~45M params)
        'name': 'proxy',
        'n_embd': 768,
        'n_head': 12,
        'n_kv_head': 4,     # r = 3 (GQA)
        'n_layer': 12,
        'batch_size': 64,
        'gradient_accumulation_steps': 4,  # effective batch 256
        'max_iters': 2000,
    },
    {   # Target model (~160M params)
        'name': 'target',
        'n_embd': 1536,
        'n_head': 24,
        'n_kv_head': 8,     # r = 3 (same GQA ratio)
        'n_layer': 12,
        'batch_size': 16,
        'gradient_accumulation_steps': 16,  # effective batch 256
        'max_iters': 2000,
    },
]

# --- LR sweep ---
# 9 learning rates log-spaced from 1e-3.5 to 1e-1.5
learning_rates = [10**p for p in np.linspace(-3.5, -1.5, 9)]

seeds = [42, 43, 44]

# --- Parameterization configs ---
param_configs = [
    {
        'label': 'SP',
        'mup': 'false',
        'impl': 'standard_param_impl',
        'base_width': None,  # no scaling
    },
    {
        'label': 'GQA-muP',
        'mup': 'true',
        'impl': 'tpv_left_impl_new_kv_2',
        'base_width': 256,  # mup_multiplier = n_embd / base_width
    },
]

configs = []
for param in param_configs:
    for model in models:
        for lr in learning_rates:
            for seed in seeds:
                conf = {}
                conf.update(shared)
                conf.update({
                    'n_embd': model['n_embd'],
                    'n_head': model['n_head'],
                    'n_kv_head': model['n_kv_head'],
                    'n_layer': model['n_layer'],
                    'batch_size': model['batch_size'],
                    'gradient_accumulation_steps': model['gradient_accumulation_steps'],
                    'max_iters': model['max_iters'],
                    'learning_rate': lr,
                    'min_lr': lr / 10,
                    'seed': seed,
                    'mup': param['mup'],
                    'impl': param['impl'],
                    'mup_multiplier': model['n_embd'] / param['base_width'] if param['base_width'] else 1,
                    'wandb_run_name': f"{param['label']}_{model['name']}_lr{lr:.2e}_s{seed}",
                })
                configs.append(conf)

if __name__ == "__main__":
    for conf in configs:
        print(json.dumps(conf))
```

IMPORTANT notes on memory for 1xA100:
- The target model (1536 wide, 12 layers) should fit on an 80GB A100 in bfloat16
  with batch_size=16
- If OOM: reduce batch_size to 8 and increase gradient_accumulation_steps to 32
- If still OOM: reduce n_layer to 8 or n_embd to 1280

### 3.5 Run the experiment

First, dry run to see how many experiments there are:
```bash
python experiments/local_orchestrator.py \
    --config_generator_file experiments/configs/mu_transfer_1gpu.py \
    --dry-run
```

This should show 2 models x 2 params x 9 LRs x 3 seeds = 108 experiments.

Time estimate on 1xA100:
- Proxy (768, 2000 iters): ~15 min each, 54 runs = ~13.5 hours
- Target (1536, 2000 iters): ~45 min each, 54 runs = ~40.5 hours
- Total: ~54 hours

If this is too long, reduce to 5 learning rates and 2 seeds:
2 x 2 x 5 x 2 = 40 experiments, ~22 hours total.

Run it:
```bash
python experiments/local_orchestrator.py \
    --config_generator_file experiments/configs/mu_transfer_1gpu.py
```

### 3.6 Create the analysis notebook

After the experiments complete, create `analysis/notebooks/mu_transfer_results.ipynb`
that:

1. Pulls data from W&B (or reads local logs)
2. For each parameterization (SP, GQA-muP) and each model size (proxy, target):
   - Plot final val loss vs learning rate (with error bars from 3 seeds)
3. Create the key figure: a 2x1 subplot showing:
   - Left: SP — LR sweep curves at proxy and target width. The optimal LR
     should SHIFT between the two widths.
   - Right: GQA-muP — LR sweep curves at proxy and target width. The optimal
     LR should be approximately the SAME at both widths. This is the mu-transfer
     signal.
4. Save the figure as `analysis/mu_transfer.pdf`

The figure should make the point visually obvious: with SP, you need to re-tune
LR when you scale up. With GQA-muP, the optimal LR found at small scale transfers.

### 3.7 Commit results

Commit the config, notebook, and figure. Do not commit W&B data or checkpoints.

---

## Phase 4: Final Checks

1. Run `git status` and make sure nothing unexpected is staged
2. Verify all Python files parse: `python -c "import ast; ..."`
3. Push to the `paper-release` branch
4. Write a brief summary of what was done and any issues encountered

---

## Troubleshooting

**OOM on target model**: Reduce batch_size, increase gradient_accumulation_steps
to maintain the same effective batch size. Or reduce block_size from 1024 to 512.

**Import errors**: The training script uses `exec(open('configurator.py').read())`.
This means you must run from the repo root directory, OR the configurator.py path
needs to be adjusted. If running via the local_orchestrator, check that the
subprocess working directory is correct.

**Noisy results**: If the mu-transfer signal is still weak:
- Increase max_iters to 4000 (train longer)
- Increase to 5 seeds
- Verify weight_decay=0.1 (too high WD can mask the signal)
- Check that mup_multiplier is computed correctly (n_embd / base_width)
- Check that the impl string matches: 'tpv_left_impl_new_kv_2' for GQA-muP

**W&B not working**: Set WANDB_API_KEY env var, or use --wandb_log=False and
analyze training logs from stdout instead.
