# Handoff: A100 Local Testing → H100 SLURM Cluster

This document describes the workflow for validating experiments on a single
A100, then handing off to a collaborator with an H100 SLURM cluster.

## The Problem

Experiment configs currently bundle two kinds of settings:
- **Science params** (model arch, LR, seeds, muP impl) — same everywhere
- **Infra params** (batch_size, n_gpus, gradient_accumulation_steps, SLURM
  settings) — depend on the hardware

We need configs to work on both a 1xA100 and an 8xH100 node without
editing the science params.

## Solution: Infra Override Files

Experiment configs define the science. Infra settings are overridden at
launch time via a hardware profile.

### Step 1: Create hardware profiles

Create `experiments/hardware/a100_1x.json`:
```json
{
    "batch_size": <from benchmark>,
    "gradient_accumulation_steps": <computed>,
    "n_gpus": 1,
    "compile": true,
    "dtype": "bfloat16"
}
```

Create `experiments/hardware/h100_8x.json`:
```json
{
    "batch_size": <from benchmark on H100>,
    "gradient_accumulation_steps": <computed>,
    "n_gpus": 8,
    "sbatch_nodes": 1,
    "sbatch_mem": 256,
    "partition": "gpu",
    "qos": "normal",
    "compile": true,
    "dtype": "bfloat16"
}
```

The `batch_size` values come from running `scripts/benchmark.py` on each
machine. The `gradient_accumulation_steps` is then chosen to hit a desired
effective batch size:

    effective_batch = batch_size * gradient_accumulation_steps * n_gpus

### Step 2: Orchestrators apply hardware overrides

Both orchestrators (`local_orchestrator.py` and `orchestrator.py`) accept
a `--hardware` flag that merges the hardware profile into each experiment
config, overriding any infra keys:

```bash
# On A100
python experiments/local_orchestrator.py \
    --config_generator_file experiments/configs/mu_transfer_1gpu.py \
    --hardware experiments/hardware/a100_1x.json

# On H100 cluster
python experiments/orchestrator.py \
    --config_generator_file experiments/configs/mu_transfer_1gpu.py \
    --hardware experiments/hardware/h100_8x.json
```

---

## Kyle's A100 Workflow

### 1. Setup & data prep
```bash
uv sync --group data --group analysis
uv run python data/prepare_openwebtext.py
```

### 2. Benchmark
```bash
uv run python scripts/benchmark.py --output experiments/hardware/a100_bench.json
```
Note the recommended batch_size, then create `experiments/hardware/a100_1x.json`.

### 3. Smoke test
Run a short training job to verify imports, model, data loading, wandb:
```bash
uv run python gqa_mup/train.py \
    --n_embd=768 --n_head=12 --n_kv_head=4 --n_layer=12 \
    --batch_size=<from benchmark> --max_iters=100 \
    --learning_rate=3e-4 --mup=True --mup_multiplier=3.0 \
    --impl=tpv_left_impl_new_kv_2 \
    --wandb_log=True --wandb_project=gqa-mup-smoke-test \
    --compile=True --device=cuda --dtype=bfloat16
```

### 4. Validate coord check
```bash
bash scripts/coord_check.sh 0
```
Check that CSV is produced in coord_check_test/.

### 5. Run a TINY mu-transfer test
Run just 2 LRs × 1 seed × proxy model only to verify the pipeline works
end-to-end and produces W&B runs you can plot:
```bash
python experiments/local_orchestrator.py \
    --config_generator_file experiments/configs/mu_transfer_1gpu.py \
    --hardware experiments/hardware/a100_1x.json \
    --max-experiments 2
```
Verify runs appear in W&B dashboard.

### 6. (Optional) Run proxy-only sweep
The proxy model (768w, ~154M params) is cheap. A full proxy LR sweep
validates that the experiment design produces clean LR-vs-loss curves:
```bash
python experiments/local_orchestrator.py \
    --config_generator_file experiments/configs/mu_transfer_1gpu.py \
    --hardware experiments/hardware/a100_1x.json \
    --max-experiments 54   # 2 params x 9 LRs x 3 seeds, proxy only
```
Eyeball the W&B results — the LR-vs-loss curves should be smooth parabolas
with clear minima. If they're too noisy, increase max_iters in the config.

### 7. Polish notebooks
See plan.md Phase 2.

### 8. Package for handoff
```bash
git add -A && git commit -m "validated on A100, ready for cluster"
git push
```

---

## Collaborator's H100 Cluster Workflow

### 1. Setup
```bash
git clone git@github.com:KyroChi/nanoGPT-maximal-paramaterizations.git
cd nanoGPT-maximal-paramaterizations
git checkout paper-release
uv sync --group data
uv run python data/prepare_openwebtext.py
export WANDB_API_KEY=<key>
```

### 2. Benchmark on H100
```bash
uv run python scripts/benchmark.py --output experiments/hardware/h100_bench.json
```
Create `experiments/hardware/h100_8x.json` with the results.

### 3. Run the full experiment
```bash
python experiments/orchestrator.py \
    --config_generator_file experiments/configs/mu_transfer_1gpu.py \
    --hardware experiments/hardware/h100_8x.json
```

### 4. Extract results
```bash
uv run python analysis/crawl_wandb.py \
    --entity <entity> --project mu-transfer-1gpu --output-dir results/
```

---

## Effective Batch Size Guidance

For mu-transfer experiments, the effective batch size should be the same
across all model sizes and hardware configs. This ensures comparable
training dynamics.

Pick one effective batch (e.g., 64 or 128 sequences = 64K–128K tokens),
then compute gradient_accumulation_steps:

    grad_accum = effective_batch / (batch_size * n_gpus)

| Hardware  | batch_size | n_gpus | grad_accum | effective_batch |
|-----------|------------|--------|------------|-----------------|
| 1xA100    | 32         | 1      | 4          | 128             |
| 1xA100    | 32         | 1      | 2          | 64              |
| 8xH100    | 32         | 8      | 1          | 256             |
| 8xH100    | 64         | 8      | 1          | 512             |

(Replace batch_size with actual benchmark results.)

The experiment configs should define `effective_batch` as the science
parameter. The hardware override then fills in the concrete batch_size
and gradient_accumulation_steps.

---

## What the Collaborator Needs to Know

1. **Read CLAUDE.md** for setup and project overview
2. **Read plan.md** for compute estimates and experiment design
3. **Run benchmark.py** to get H100-specific batch sizes
4. **Check W&B** — all runs log to W&B. The wandb_project in the config
   determines where runs go. Coordinate on the W&B entity/project name.
5. **Don't change science params** — model arch, LR sweep range, seeds,
   and muP impl should stay as Kyle validated on A100.
6. **Do change infra params** — batch_size, n_gpus, grad_accum, SLURM
   partition/qos/mem are hardware-specific and should be adjusted.
