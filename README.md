# GQA-muP

Code for the paper: **GQA-muP: The Maximal Parameterization Update for Grouped Query Attention and Fully Sharded Data Parallel**

[OpenReview](https://openreview.net/forum?id=UJB2uOS9MR)

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# For analysis/notebooks
uv sync --group analysis

# For data preparation
uv sync --group data
```

### Data

Prepare OpenWebText:

```bash
uv run python data/prepare_openwebtext.py
```

## Project Structure

```
gqa_mup/                    # Core library
  model.py                  # GPT with Grouped Query Attention + RoPE
  mup.py                    # muP scaling rules (impl_dict)
  train.py                  # Training loop (DDP, coordinate checking)
  coord_check.py            # Coordinate checking utilities
  configurator.py           # Config override system
  indexed_dataset.py        # Data loading (megatron IndexedDataset stub)

experiments/                # Experiment pipeline
  orchestrator.py           # SLURM job orchestrator
  local_orchestrator.py     # Local sequential runner (no SLURM)
  run_experiment.py         # Single experiment launcher
  configs/                  # Experiment config generators

analysis/                   # Results
  crawl_wandb.py            # W&B data extraction
  notebooks/                # Analysis & figure notebooks

scripts/                    # Shell scripts
  coord_check.sh            # Coordinate checking (SLURM or local)
  job.sh                    # SLURM job template

data/                       # Dataset preparation
  prepare_openwebtext.py
```

## Running Experiments

### With SLURM

```bash
# Run a single experiment config
python experiments/orchestrator.py --config_generator_file experiments/configs/width_only.py

# Dry run (see what would be submitted)
python experiments/orchestrator.py --config_generator_file experiments/configs/width_only.py --dry-run
```

### Locally (no SLURM)

```bash
# Run experiments sequentially on local GPU
python experiments/local_orchestrator.py --config_generator_file experiments/configs/width_only.py

# Dry run
python experiments/local_orchestrator.py --config_generator_file experiments/configs/width_only.py --dry-run

# Limit number of experiments
python experiments/local_orchestrator.py --config_generator_file experiments/configs/width_only.py --max-experiments 3
```

### Coordinate Checking

```bash
# On SLURM
sbatch --array=0-5 scripts/coord_check.sh

# Locally (run index 0)
bash scripts/coord_check.sh 0
```

## Extracting Results

```bash
python analysis/crawl_wandb.py --entity <your-entity> --project <project-name> --output-dir results/
```

## Key Experiment Configs

| Config | What it tests |
|--------|--------------|
| `width_only.py` | Width sweep with LR transfer |
| `depth_only.py` | Depth sweep with LR transfer |
| `head_size.py` | Head dimension ablations |
| `head_size_transfer.py` | Head size HP transfer |
| `r_ablations.py` | KV repetition ratio ablations |
| `gqa_transfer_2.py` | GQA muP transfer test |
| `mu_transfer_all.py` | Full mu-transfer experiment |
| `scaling_law_configs.py` | Scaling law measurements |
| `ablations_baselines.py` | Standard parameterization baselines |

## Citation

```bibtex
@inproceedings{
  title={GQA-muP: The Maximal Parameterization Update for Grouped Query Attention and Fully Sharded Data Parallel},
  booktitle={},
  year={2025},
  url={https://openreview.net/forum?id=UJB2uOS9MR}
}
```

## License

MIT
