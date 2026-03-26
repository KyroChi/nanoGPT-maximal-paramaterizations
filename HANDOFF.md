# Cluster Handoff: Instructions for H+M

This document is everything you need to run the GQA-muP experiments on
your SLURM cluster. Kyle has validated the code on a single H100 — you
should not need to debug anything. If something fails, report the error
to Kyle rather than trying to fix it.

## Quick Start

```bash
# 1. Clone
git clone git@github.com:KyroChi/nanoGPT-maximal-paramaterizations.git
cd nanoGPT-maximal-paramaterizations
git checkout paper-release

# 2. Install (requires uv: https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra data

# 3. Prepare data (~1-2 hours, ~54GB download, ~17GB on disk)
uv run python data/prepare_openwebtext.py

# 4. Set up W&B
export WANDB_API_KEY=<your-key>

# 5. Benchmark your hardware
uv run python scripts/benchmark.py --output benchmark_results.json

# 6. Run experiments (see below)
```

## Prerequisites

- Python 3.10+
- NVIDIA GPUs with CUDA 12.x
- ~25GB disk for data (train.bin + val.bin)
- ~54GB temporary disk for HuggingFace cache during data prep
- W&B account (all runs log to Weights & Biases)
- `jq` installed (used by SLURM orchestrator for JSON parsing)

## Experiments to Run

### Experiment 2: GQA Ablation

Varies the KV repetition factor r at fixed width 1536, comparing muP
without GQA correction (`mup_no_kv`) vs our GQA-muP (`gqa_mup`).

462 runs. Each run is ~1 hour on 1xH100. With cluster parallelism this
should complete in minutes.

```bash
# Dry run — see what jobs would be submitted
python experiments/orchestrator.py \
    --config_generator_file experiments/configs/r_ablations_2.py \
    --dry-run

# Submit
python experiments/orchestrator.py \
    --config_generator_file experiments/configs/r_ablations_2.py \
    --max_concurrent 50
```

**W&B project name:** Set in the config file (currently
`ablate-gqa-repetition-kyle-impl-6`). Change the `WANDB_PROJECT`
variable at the top of `experiments/configs/r_ablations_2.py` if you
want a different project name.

### Experiment 3: Large-Scale Transfer

3-scale width transfer (256w → 1024w → 4096w) comparing SP vs GQA-muP.

96 runs. Largest model is 945M params (4096w, 3 layers).

```bash
# Dry run
python experiments/orchestrator.py \
    --config_generator_file experiments/configs/reviewer_transfer.py \
    --dry-run

# Submit
python experiments/orchestrator.py \
    --config_generator_file experiments/configs/reviewer_transfer.py \
    --max_concurrent 30
```

**W&B project name:** `gqa-mup-reviewer-transfer`

## Hardware Overrides

The experiment configs have default batch_size and n_gpus values that
may not match your cluster. You can override these without editing the
configs:

1. Create a JSON file, e.g. `experiments/hardware/cluster.json`:
```json
{
    "batch_size": 64,
    "gradient_accumulation_steps": 1,
    "n_gpus": 8,
    "sbatch_nodes": 1,
    "sbatch_mem": 256,
    "partition": "your-partition",
    "qos": "your-qos",
    "compile": "true",
    "dtype": "bfloat16"
}
```

2. Pass it to the orchestrator:
```bash
python experiments/orchestrator.py \
    --config_generator_file experiments/configs/reviewer_transfer.py \
    --hardware experiments/hardware/cluster.json \
    --max_concurrent 30
```

The hardware override merges into every experiment config, overwriting
any matching keys. Science parameters (model arch, LR, seeds, impl) are
not affected.

**Important:** If you change `batch_size` or `gradient_accumulation_steps`
or `n_gpus`, the effective batch size changes, which changes how many
iterations are needed for the same token budget. The `reviewer_transfer.py`
config computes `max_iters` from the token budget automatically, so
overriding these is safe. The `r_ablations_2.py` config has hardcoded
`max_iters` — if you change batch size there, the token budget will change.

## Extracting Results

After runs complete:

```bash
uv run python analysis/crawl_wandb.py \
    --entity <your-wandb-entity> \
    --project gqa-mup-reviewer-transfer \
    --output-dir results/reviewer_transfer/

uv run python analysis/crawl_wandb.py \
    --entity <your-wandb-entity> \
    --project ablate-gqa-repetition-kyle-impl-6 \
    --output-dir results/gqa_ablation/
```

## Troubleshooting

**SLURM job fails immediately:** Check that `partition` and `qos` match
your cluster. Use `--hardware` override or edit the config file.

**"jq: command not found":** Install jq: `sudo apt install jq` or
`conda install -c conda-forge jq`.

**Data not found:** The training script expects `data/openwebtext/train.bin`
and `data/openwebtext/val.bin` (note: inside a subdirectory). If you ran
`prepare_openwebtext.py` from the repo root, the files will be at
`data/train.bin` and `data/val.bin`. Either move them or create a symlink:
```bash
mkdir -p data/openwebtext
ln -s ../train.bin data/openwebtext/train.bin
ln -s ../val.bin data/openwebtext/val.bin
```

**OOM:** Reduce `batch_size` in your hardware override and increase
`gradient_accumulation_steps` proportionally to keep the same effective
batch.

**Any other error:** Do not debug. Send the full error message and the
command you ran to Kyle.

## Do Not Modify

- Anything in `gqa_mup/` (model, training loop, muP implementations)
- LR ranges, seed lists, or impl names in experiment configs
- The `--impl` or `--mup` or `--mup_multiplier` values

You may modify:
- `WANDB_PROJECT` in config files
- Hardware settings via `--hardware` override
- `--max_concurrent` for SLURM job concurrency
