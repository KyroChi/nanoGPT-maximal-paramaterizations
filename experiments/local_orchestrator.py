"""
Local orchestrator for running experiments sequentially without SLURM.

Reads a config generator script that outputs one JSON config per line,
then runs each experiment locally as a subprocess.
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

args = parser.parse_args()


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


def config_to_args(config_json):
    """Convert a JSON config string to a list of CLI arguments for train.py."""
    config = json.loads(config_json)
    cli_args = []
    for key, value in config.items():
        if value is True:
            cli_args.append(f"--{key}")
        elif value is False or value is None:
            continue  # skip false flags and null values
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
            desc = ", ".join(f"{k}={v}" for k, v in list(config.items())[:4])
            if len(config) > 4:
                desc += ", ..."
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
