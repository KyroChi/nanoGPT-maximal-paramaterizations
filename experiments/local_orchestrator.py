"""
Local orchestrator for running experiments sequentially without SLURM.

Reads a config generator script that outputs one JSON config per line,
then runs each experiment locally as a subprocess.

Hardware overrides (--hardware) let you separate science params from infra
params. Pass a JSON file with batch_size, gradient_accumulation_steps, etc.
to override those keys in every experiment config.
"""

import argparse
import json
import subprocess
import sys

TRAINING_SCRIPT = "gqa_mup/train.py"

parser = argparse.ArgumentParser(description="Run experiments locally (no SLURM).")
parser.add_argument('--config_generator_file', type=str, required=True,
                    help='Path to a Python script that prints one JSON config per line.')
parser.add_argument('--dry-run', action='store_true',
                    help='Print commands that would be run without executing them.')
parser.add_argument('--max-experiments', type=int, default=None,
                    help='Maximum number of experiments to run (default: all).')
parser.add_argument('--hardware', type=str, default=None,
                    help='Path to a JSON file with hardware-specific overrides '
                         '(batch_size, gradient_accumulation_steps, dtype, etc.)')

args = parser.parse_args()

# Load hardware overrides if provided
hardware_overrides = {}
if args.hardware:
    with open(args.hardware) as f:
        hardware_overrides = json.load(f)
    print(f"Hardware overrides from {args.hardware}: {hardware_overrides}")


def run_config_generator(config_generator_file):
    """Run the config generator script and return its stdout."""
    command = ['python', config_generator_file]
    if args.dry_run:
        command.append('--dry-run')
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running config generator: {e}")
        print(f"stderr: {e.stderr}")
        return None


def apply_overrides(config_json, overrides):
    """Merge hardware overrides into a config, returning updated JSON string."""
    if not overrides:
        return config_json
    config = json.loads(config_json)
    config.update(overrides)
    return json.dumps(config)


def config_to_args(config_json):
    """Convert a JSON config string to a list of CLI arguments for train.py."""
    config = json.loads(config_json)
    cli_args = []
    # Skip keys that are SLURM-only (not understood by train.py)
    slurm_keys = {'n_gpus', 'sbatch_nodes', 'sbatch_mem', 'sbatch_timeout',
                  'partition', 'qos', 'reservation', 'sbatch_exclusive',
                  'cpus_per_task', 'sbatch_logging_dir', 'log_wandb', 'name'}
    for key, value in config.items():
        if key in slurm_keys:
            continue
        # Remap log_wandb -> wandb_log for train.py compatibility
        if key == 'log_wandb':
            key = 'wandb_log'
        if value is True or value == 'true':
            cli_args.append(f"--{key}=True")
        elif value is False or value == 'false':
            cli_args.append(f"--{key}=False")
        elif value is None:
            continue
        else:
            cli_args.append(f"--{key}={value}")
    return cli_args


def main():
    configs_str = run_config_generator(args.config_generator_file)
    if configs_str is None:
        print("Failed to generate configurations. Exiting.")
        sys.exit(1)

    config_lines = [line.strip() for line in configs_str.split('\n') if line.strip()]
    total = len(config_lines)

    # Apply hardware overrides to each config
    config_lines = [apply_overrides(c, hardware_overrides) for c in config_lines]

    if args.max_experiments is not None:
        config_lines = config_lines[:args.max_experiments]

    run_count = len(config_lines)
    print(f"Will run {run_count}/{total} experiments.")

    for i, config_json in enumerate(config_lines, start=1):
        cli_args = config_to_args(config_json)
        command = ['python', TRAINING_SCRIPT] + cli_args

        # Build a short description from the config for progress display
        try:
            config = json.loads(config_json)
            show_keys = ['n_embd', 'n_kv_head', 'learning_rate', 'impl', 'seed']
            desc = ", ".join(f"{k}={config[k]}" for k in show_keys if k in config)
        except json.JSONDecodeError:
            desc = config_json[:80]

        print(f"\n{'='*60}")
        print(f"Running experiment {i}/{run_count}: {desc}")
        print(f"{'='*60}")

        if args.dry_run:
            print(f"  Command: {' '.join(command)}")
            continue

        try:
            result = subprocess.run(command)
            if result.returncode != 0:
                print(f"WARNING: Experiment {i}/{run_count} exited with code {result.returncode}")
            else:
                print(f"Experiment {i}/{run_count} completed successfully.")
        except KeyboardInterrupt:
            print(f"\nInterrupted during experiment {i}/{run_count}. Stopping.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Experiment {i}/{run_count} failed: {e}")

    print(f"\nDone. Ran {run_count} experiment(s).")


if __name__ == '__main__':
    main()
