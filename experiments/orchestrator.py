"""
Orchestrator for submitting experiment configurations as SLURM array jobs.

Reads a config generator script that outputs one JSON config per line,
then submits them as a SLURM job array with configurable concurrency.
"""

import argparse
import datetime
import os
import subprocess

COMMAND_FILE = "experiments/run_experiment.py"

parser = argparse.ArgumentParser(description="Submit experiment configs as SLURM array jobs.")
parser.add_argument('--config_generator_file', type=str, required=True,
                    help='Path to a Python script that prints one JSON config per line.')
parser.add_argument('--max_concurrent', type=int, default=30,
                    help='Maximum number of concurrent SLURM array tasks.')
parser.add_argument('--dry-run', action='store_true',
                    help='Pass --dry-run to the config generator and print the script without submitting.')

args = parser.parse_args()

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_logging_dir = "experiments/slurm_logs"
logging_dir = f"{base_logging_dir}/{now}"
orchestrator_dir = f"{logging_dir}/orchestrator"

os.makedirs(base_logging_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(orchestrator_dir, exist_ok=True)


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


configs = run_config_generator(args.config_generator_file)
if configs is None:
    print("Failed to generate configurations. Exiting.")
    exit(1)

num_experiments = len(configs.split('\n'))
print(f"Generated {num_experiments} experiment configurations.")

sbatch_headers = f"""#!/bin/bash

#SBATCH --array=0-{num_experiments - 1}%{min(args.max_concurrent, num_experiments)}
#SBATCH --job-name=kyle_orchestrator
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --output={orchestrator_dir}/%A_%a.out
#SBATCH --error={orchestrator_dir}/%A_%a.err
#SBATCH --mem=8G
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --distribution=pack

"""

# Build the CONFIGS bash array
parsed_config_str = ""
for config in configs.split('\n'):
    parsed_config_str += f"    '{config.strip()}'\n"
parsed_config_str = parsed_config_str.rstrip('\n')

config_str = f"""CONFIGS=(
{parsed_config_str}
)

"""

# Parse JSON config into CLI arguments
parsing_str = f"""CONFIG_JSON="${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}"
# Parse the individual configs
ARGS=""
while IFS='=' read -r key value;
do
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)

    if [ -z "$key" ];
    then
        continue
    fi

    key="${{key%\\"}}"
    key="${{key#\\"}}"

    if [ "$value" == "true" ];
    then
        ARGS+=" --$key"
    elif [ "$value" == "false" ];
    then
        true # no-op
    elif [ "$value" == "null" ];
    then
        true # no-op
    else
        ESCAPED_VALUE=$(printf %q "$value")
        ARGS+=" --$key ${{ESCAPED_VALUE}}"
    fi
done < <(echo "$CONFIG_JSON" | jq -r 'to_entries[] | .key + "=" + (.value | @json)')

ARGS+=" --sbatch_logging_dir {logging_dir}"
ARGS+=" --out_dir {logging_dir}/ckpts/"

"""

command_str = f"""echo $ARGS
python {COMMAND_FILE} $ARGS &

PID=$!
wait $PID
"""

shell_script = sbatch_headers + config_str + parsing_str + command_str

if args.dry_run:
    print("--- Generated SLURM script (dry run) ---")
    print(shell_script)
    print("--- End of script ---")
else:
    try:
        process = subprocess.run(
            ['sbatch'], input=shell_script, text=True, capture_output=True, check=True
        )
        print(f"Job submitted successfully: {process.stdout.strip()}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr.strip()}", flush=True)
        exit(1)
