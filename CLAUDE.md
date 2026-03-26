# GQA-muP

Paper: https://openreview.net/forum?id=UJB2uOS9MR

## Setup from scratch

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install

```bash
git clone git@github.com:KyroChi/nanoGPT-maximal-paramaterizations.git
cd nanoGPT-maximal-paramaterizations
git checkout paper-release
uv sync
```

For analysis/notebooks: `uv sync --extra analysis`
For data prep (huggingface datasets): `uv sync --extra data`

### 3. Prepare OpenWebText

This downloads ~54GB to the HuggingFace cache and produces two binary files in `data/`:

```bash
uv run python data/prepare_openwebtext.py
```

Output:
- `data/train.bin` (~17GB, ~9B tokens)
- `data/val.bin` (~8.5MB, ~4M tokens)

### 4. Set up W&B

```bash
export WANDB_API_KEY=<your-key>
uv run wandb login
```

### 5. Benchmark batch size

Before running experiments, benchmark to find the optimal microbatch size for your GPU. This sweeps batch sizes across model configs and recommends the one that maximizes throughput (tokens/sec):

```bash
uv run python scripts/benchmark.py
```

This benchmarks the proxy (768w) and target (1536w) models by default. Output includes actual tokens/sec, MFU%, peak memory, and a recommended unified batch size.

Options:
```bash
# Benchmark a specific model
uv run python scripts/benchmark.py --n_embd 1536 --n_layer 12

# Custom batch sizes
uv run python scripts/benchmark.py --batch_sizes 4,8,16,32,48,64,96,128

# Set target tokens for time estimates (default 500M)
uv run python scripts/benchmark.py --target_tokens 1_000_000_000

# Save results to JSON
uv run python scripts/benchmark.py --output benchmark_results.json

# Disable torch.compile (faster startup, lower throughput)
uv run python scripts/benchmark.py --no-compile
```

Use the recommended `batch_size` in your experiment configs. Set `gradient_accumulation_steps` to achieve whatever effective batch you want — throughput (tokens/sec) is the same regardless of grad accum.

### 6. Run a training job

Single GPU:
```bash
uv run python gqa_mup/train.py --wandb_log=True --wandb_project=gqa-mup
```

Multi-GPU (DDP):
```bash
uv run torchrun --standalone --nproc_per_node=4 gqa_mup/train.py
```

### 7. Run experiments

With SLURM:
```bash
python experiments/orchestrator.py --config_generator_file experiments/configs/width_only.py
```

Locally (sequential, no SLURM):
```bash
python experiments/local_orchestrator.py --config_generator_file experiments/configs/width_only.py
```

Dry run (prints commands without executing):
```bash
python experiments/local_orchestrator.py --config_generator_file experiments/configs/width_only.py --dry-run
```

### 8. Coordinate checking

```bash
# Locally (run experiment index 0)
bash scripts/coord_check.sh 0

# On SLURM
sbatch --array=0-5 scripts/coord_check.sh
```

### 9. Extract W&B results

```bash
uv run python analysis/crawl_wandb.py --entity <your-entity> --project <project-name> --output-dir results/
```

## Project layout

```
gqa_mup/                    # Core library
  model.py                  # GPT with GQA + RoPE
  mup.py                    # muP scaling rules (impl_dict)
  train.py                  # Training loop (DDP, coord checking, wandb)
  coord_check.py            # Coordinate checking utilities
  configurator.py           # Config override system (exec-based, nanoGPT style)
  indexed_dataset.py        # Stub for megatron IndexedDataset (SlimPajama only)

experiments/                # Experiment pipeline
  orchestrator.py           # SLURM batch orchestrator
  local_orchestrator.py     # Local sequential runner (no SLURM)
  run_experiment.py         # Single experiment SLURM launcher
  configs/                  # Experiment config generators (26 configs)

analysis/                   # Results extraction and figures
  crawl_wandb.py            # W&B data extraction
  notebooks/                # 6 analysis/figure notebooks

scripts/                    # Shell scripts and utilities
  benchmark.py              # GPU batch size tuning (run before experiments)
  coord_check.sh            # Coordinate checking (SLURM or local)
  job.sh                    # SLURM job template
```

## Architecture notes

- **Config system**: Uses nanoGPT's `exec(open('configurator.py').read())` pattern. Config vars are module-level globals overridden via CLI args like `--n_embd=512`.
- **muP implementations**: `gqa_mup/mup.py` contains `impl_dict` mapping string names to dicts of scaling rules. Each impl dict has keys for `embedding`, `hidden`, `kv_layer`, `unembedding`, `normalization`, `attention_scale`, `depth_scale`. The default is `xllm_impl`.
- **Experiment configs**: Each file in `experiments/configs/` is a Python script that prints JSON configs to stdout. The orchestrator runs these scripts, parses the output, and launches jobs.
- **Data**: Training uses numpy memmap binary files (train.bin/val.bin) for OpenWebText. SlimPajama support exists but requires megatron-core (not installed by default).

## PyTorch CUDA

The `pyproject.toml` uses `extra-index-url` for CUDA 12.4 wheels. If your cluster uses a different CUDA version, update the URL in `[tool.uv]`:
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`
- CUDA 12.4: `https://download.pytorch.org/whl/cu124`
- CPU only: `https://download.pytorch.org/whl/cpu`
