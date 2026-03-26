"""
Large-scale mu-transfer experiment for reviewer response.

Shows that GQA-muP transfers optimal LR across a 16x width gap (28M → 945M).
Compares against standard parameterization (SP) where LR does NOT transfer.

3 scales:
  - Proxy:  256w,  3L,  28M params  (20 TPP, full LR sweep)
  - Mid:   1024w,  3L, 137M params  (10 TPP, full LR sweep)
  - Target: 4096w, 3L, 945M params  ( 3 TPP, full LR sweep)

Total runs: 96 (= 3 scales × 2 params × {9,7,7} LRs × {3,2,1} seeds)

Cluster cost (8xH100, 12 concurrent jobs): ~3 hours wall clock.
Single H100: ~24 hours sequential.

Usage:
    # Dry run
    python experiments/local_orchestrator.py \\
        --config_generator_file experiments/configs/reviewer_transfer.py --dry-run

    # SLURM cluster
    python experiments/orchestrator.py \\
        --config_generator_file experiments/configs/reviewer_transfer.py --max_concurrent 16
"""
import json
import numpy as np
from copy import deepcopy

# ============================================================
# WANDB CONFIG — collaborator: change entity if needed
# ============================================================
WANDB_PROJECT = 'gqa-mup-reviewer-transfer'

# ============================================================
# SHARED TRAINING CONFIG (science params — do not change)
# ============================================================
shared = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 200,
    'eval_iters': 20,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': 0.1,
    'warmup_iters': 200,
    'dataset': 'openwebtext',
    'block_size': 1024,
    'compile': 'true',
    'dtype': 'bfloat16',
    'bias': 'false',
    'dropout': 0.0,
    'init_std': 0.02,
    'normalization': 'LayerNorm',
}

# ============================================================
# MODEL SCALES
# ============================================================
# All models: n_layer=3, head_dim=64, GQA ratio r≈4
# base_width=256 for mup_multiplier calculation
BASE_WIDTH = 256

models = [
    {
        'name': 'proxy',
        'n_embd': 256,
        'n_head': 4,
        'n_kv_head': 2,
        'n_layer': 3,
        # Infra defaults (overridden by --hardware if provided)
        'batch_size': 128,
        'gradient_accumulation_steps': 1,
        'n_gpus': 1,
    },
    {
        'name': 'mid',
        'n_embd': 1024,
        'n_head': 16,
        'n_kv_head': 4,
        'n_layer': 3,
        'batch_size': 64,
        'gradient_accumulation_steps': 2,
        'n_gpus': 1,
    },
    {
        'name': 'target',
        'n_embd': 4096,
        'n_head': 64,
        'n_kv_head': 16,
        'n_layer': 3,
        'batch_size': 8,
        'gradient_accumulation_steps': 16,
        'n_gpus': 8,
        'sbatch_mem': 256,
    },
]

# ============================================================
# TOKENS PER PARAMETER (training length per scale)
# ============================================================
TPP = {
    'proxy': 20,
    'mid': 10,
    'target': 3,
}

# ============================================================
# LR SWEEP (log-uniform)
# ============================================================
LR_GRID = {
    'proxy': [10**p for p in np.linspace(-3.5, -1.0, 9)],
    'mid':   [10**p for p in np.linspace(-3.5, -1.0, 7)],
    'target': [10**p for p in np.linspace(-3.5, -1.0, 7)],
}

SEEDS = {
    'proxy': [42, 43, 44],
    'mid':   [42, 43],
    'target': [42],
}

# ============================================================
# PARAMETERIZATION CONFIGS
# ============================================================
param_configs = [
    {
        'label': 'SP',
        'mup': 'false',
        'impl': 'sp',
    },
    {
        'label': 'GQA-muP',
        'mup': 'true',
        'impl': 'gqa_mup',
    },
]


def compute_max_iters(n_params, tpp, batch_size, grad_accum, n_gpus, block_size=1024):
    """Compute max_iters from tokens-per-parameter target."""
    total_tokens = n_params * tpp
    tokens_per_iter = batch_size * grad_accum * n_gpus * block_size
    return max(100, int(total_tokens / tokens_per_iter))


def approx_params(n_embd, n_layer, n_head, n_kv_head):
    """Rough param count for TPP calculation."""
    hd = n_embd // n_head
    pl = 2*n_embd**2 + 2*n_embd*(n_kv_head*hd) + 2*n_embd*4*n_embd
    nl = 2*50304*n_embd + 1024*n_embd
    return nl + n_layer * pl


# ============================================================
# GENERATE CONFIGS
# ============================================================
configs = []

for param in param_configs:
    for model in models:
        scale = model['name']
        n_params = approx_params(model['n_embd'], model['n_layer'],
                                  model['n_head'], model['n_kv_head'])

        max_iters = compute_max_iters(
            n_params, TPP[scale],
            model['batch_size'], model['gradient_accumulation_steps'],
            model['n_gpus'],
        )

        for lr in LR_GRID[scale]:
            for seed in SEEDS[scale]:
                conf = {}
                conf.update(shared)
                conf.update({
                    'n_embd': model['n_embd'],
                    'n_head': model['n_head'],
                    'n_kv_head': model['n_kv_head'],
                    'n_layer': model['n_layer'],
                    'batch_size': model['batch_size'],
                    'gradient_accumulation_steps': model['gradient_accumulation_steps'],
                    'n_gpus': model.get('n_gpus', 1),
                    'max_iters': max_iters,
                    'learning_rate': lr,
                    'min_lr': lr / 10,
                    'seed': seed,
                    'mup': param['mup'],
                    'impl': param['impl'],
                    'mup_multiplier': model['n_embd'] / BASE_WIDTH if param['mup'] == 'true' else 1,
                    'wandb_run_name': f"{param['label']}_{scale}_{model['n_embd']}w_lr{lr:.2e}_s{seed}",
                })
                if 'sbatch_mem' in model:
                    conf['sbatch_mem'] = model['sbatch_mem']
                configs.append(conf)

if __name__ == "__main__":
    import sys

    # Print summary to stderr so it doesn't pollute config output
    n_proxy = sum(1 for c in configs if c['n_embd'] == 256)
    n_mid = sum(1 for c in configs if c['n_embd'] == 1024)
    n_target = sum(1 for c in configs if c['n_embd'] == 4096)
    print(f"Generated {len(configs)} configs: "
          f"{n_proxy} proxy, {n_mid} mid, {n_target} target", file=sys.stderr)

    for conf in configs:
        print(json.dumps(conf))
