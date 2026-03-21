"""
Script to handle command line arguments for training a model via SLURM.
This allows one to fully parameterize and execute training runs from the
command line. It constructs an sbatch script and submits it, requesting
GPUs for larger runs to get higher MFUs.
"""

import argparse
import datetime
import os
import subprocess

TRAINING_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gqa_mup', 'train.py')


def gpt_params(seq_len, vocab_size, d_model, num_heads, num_layers):
    """Given GPT config, calculate total number of parameters."""
    ffw_size = 4 * d_model
    # token and position embeddings
    embeddings = d_model * vocab_size + d_model * seq_len
    # transformer blocks
    attention = 3 * d_model**2 + 3 * d_model
    attproj = d_model**2 + d_model
    ffw = d_model * ffw_size + ffw_size
    ffwproj = ffw_size * d_model + d_model
    layernorms = 2 * 2 * d_model
    # dense
    ln_f = 2 * d_model
    dense = d_model * vocab_size  # no bias
    total_params = num_layers * (attention + attproj + ffw + ffwproj + layernorms) + ln_f + dense
    return total_params


now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser(description="Submit a training run via SLURM.")

# Sbatch arguments
parser.add_argument('--sbatch_timeout', type=str, default='50:00:00')
parser.add_argument('--sbatch_nodes', type=int, default=1)
parser.add_argument('--sbatch_exclusive', action='store_true')
parser.add_argument('--n_gpus', type=int, default=8)
parser.add_argument('--cpus-per-task', type=int, default=16)
parser.add_argument('--sbatch_logging_dir', type=str, default='slurm_logs')
parser.add_argument('--sbatch_mem', type=int, default=50)
parser.add_argument('--partition', type=str, default='lowprio')
parser.add_argument('--qos', type=str, default='lowprio')
parser.add_argument('--reservation', type=str, default=None)

# Model testbed arguments
parser.add_argument('--out_dir', type=str, default=f'model_training/{now}')
parser.add_argument('--log_wandb', action='store_true')
parser.add_argument('--wandb_run_name', type=str, default='gpt')
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--backend', type=str, default='nccl')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--compile', action='store_true')
parser.add_argument('--coord_check', action='store_true')

# Evaluation parameters
parser.add_argument('--eval_interval', type=int, default=10000000000)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--avg_interval', type=int, default=30)
parser.add_argument('--eval_iters', type=int, default=300)
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--enable_checkpointing', action='store_true')

# Initialization and dataset
parser.add_argument('--init_from', type=str, default='scratch')
parser.add_argument('--dataset', type=str, default='openwebtext')
parser.add_argument('--block_size', type=int, default=1024)

# Model dynamics arguments
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--eps', type=float, default=1e-12)
parser.add_argument('--max_iters', type=int, default=None)
parser.add_argument('--decay_profile', type=str, default='cosine')
parser.add_argument('--lr_decay_iters', type=int, default=None)
parser.add_argument('--cooldown_iters', type=int, default=1000)
parser.add_argument('--tpp', type=int, default=5)
parser.add_argument('--warmup_iters', type=int, default=None)
parser.add_argument('--anneal_wd', action='store_true')
parser.add_argument('--min_wd', type=float, default=0.0)
parser.add_argument('--wd_warmup_iters', type=int, default=1000)
parser.add_argument('--wd_anneal_iters', type=int, default=1000)
parser.add_argument('--adaptive_optimizer', action='store_true')

# Model architecture parameters
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--n_head', type=int, default=16)
parser.add_argument('--n_kv_head', type=int, default=16)
parser.add_argument('--n_embd', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--init_std', type=float, default=0.02)
parser.add_argument('--learning_rate', type=float, default=0.000646)
parser.add_argument('--min_lr', type=float, default=0.0000646)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=42)

# muP parameters
parser.add_argument('--mup', action='store_true', default=False)
parser.add_argument('--mup_multiplier', type=float, default=1)
parser.add_argument('--complete_p_layers', action='store_true', default=False)

# Normalization and implementation
parser.add_argument('--normalization', type=str, default='RMSNorm')
parser.add_argument('--q_prelayer_normalization', type=str, default='NoNorm')
parser.add_argument('--k_prelayer_normalization', type=str, default='NoNorm')
parser.add_argument('--impl', type=str, default='tpv_left_impl')

# Optimizer parameters
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.95)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--decay_lr', action='store_true')

# FSDP
parser.add_argument('--enable_fsdp', action='store_true', default=False)

# RoPE (Rotary Position Embedding)
parser.add_argument('--use_rope', action='store_true', default=False)
parser.add_argument('--rope_theta', type=float, default=10000.0)

args = parser.parse_args()

os.makedirs(args.sbatch_logging_dir, exist_ok=True)
os.makedirs(args.out_dir, exist_ok=True)

# Compute max_iters from tokens-per-parameter (tpp) if not set
if args.max_iters is None:
    n_params = gpt_params(
        seq_len=args.block_size,
        vocab_size=50257,
        d_model=args.n_embd,
        num_heads=args.n_head,
        num_layers=args.n_layer
    )
    args.max_iters = args.tpp * int(n_params / (args.batch_size * 979))

if args.lr_decay_iters is None:
    args.lr_decay_iters = args.max_iters

if args.warmup_iters is None:
    args.warmup_iters = min(int(0.1 * args.max_iters), 1000)

# Exclusive flag grants exclusive access to the entire node
exclusive = '#SBATCH --exclusive' if args.sbatch_exclusive else ''

if args.n_gpus > 8:
    raise ValueError("n_gpus is a per-node value and cannot exceed 8.")

# Determine distributed training setup
if args.n_gpus > 0:
    dist_args = f"""
# --- PyTorch DDP Environment Variables ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"

NNODES=$SLURM_JOB_NUM_NODES

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export TRITON_CACHE_DIR="/tmp/triton-cache"

echo Node IP: $head_node_ip
echo $SLURM_JOB_NODELIST
export LOGLEVEL=INFO

DISTRIBUTED_ARGS=(
    --nproc_per_node={args.n_gpus}
    --nnodes=$NNODES
    --rdzv_id $RANDOM-$USER
    --rdzv_backend c10d
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
)
"""
else:
    dist_args = ""

# Construct the SLURM batch script
shell_script = f"""#!/bin/bash
#SBATCH --time={args.sbatch_timeout}
#SBATCH --nodes={args.sbatch_nodes}
{exclusive}
#SBATCH --gres=gpu:{args.n_gpus}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.cpus_per_task}

#SBATCH --output={args.sbatch_logging_dir}/%j.out
#SBATCH --error={args.sbatch_logging_dir}/%j.err
#SBATCH --mem={args.sbatch_mem}G
{f'#SBATCH --partition={args.partition}' if args.partition is not None else ''}
{f'#SBATCH --qos={args.qos}' if args.qos is not None else ''}
{f'#SBATCH --reservation={args.reservation}' if args.reservation is not None else ''}
#SBATCH --distribution=pack

{dist_args}

TRAINING_ARGS=(
    --out_dir="{args.out_dir}"
    --wandb_log={args.log_wandb}
    --wandb_project='{args.wandb_project}'
    --wandb_run_name='{args.wandb_run_name}'
    --eval_interval={args.eval_interval}
    --log_interval={args.log_interval}
    --avg_interval={args.avg_interval}
    --eval_iters={args.eval_iters}
    --eval_only={args.eval_only}
    --enable_checkpointing={args.enable_checkpointing}
    --init_from='{args.init_from}'
    --dataset='{args.dataset}'
    --gradient_accumulation_steps={args.gradient_accumulation_steps}
    --batch_size={args.batch_size}
    --block_size={args.block_size}
    --n_layer={args.n_layer}
    --n_head={args.n_head}
    --n_kv_head={args.n_kv_head}
    --n_embd={args.n_embd}
    --dropout={args.dropout}
    --bias={args.bias}
    --init_std={args.init_std}
    --learning_rate={args.learning_rate}
    --min_lr={args.min_lr}
    --max_iters={args.max_iters}
    --anneal_wd={args.anneal_wd}
    --min_wd={args.min_wd}
    --lr_decay_iters={args.lr_decay_iters}
    --warmup_iters={args.warmup_iters}
    --weight_decay={args.weight_decay}
    --beta1={args.beta1}
    --beta2={args.beta2}
    --grad_clip={args.grad_clip}
    --decay_lr={args.decay_lr}
    --complete_p_layers={args.complete_p_layers}
    --eps={args.eps}
    --mup={args.mup}
    --mup_multiplier={args.mup_multiplier}
    --seed={args.seed}
    --backend='{args.backend}'
    --device='{args.device}'
    --dtype='{args.dtype}'
    --compile={args.compile}
    --enable_fsdp={args.enable_fsdp}
    --coord_check={args.coord_check}
    --normalization='{args.normalization}'
    --q_prelayer_normalization='{args.q_prelayer_normalization}'
    --k_prelayer_normalization='{args.k_prelayer_normalization}'
    --impl='{args.impl}'
    --decay_profile='{args.decay_profile}'
    --cooldown_iters={args.cooldown_iters}
    --slurm_job_id="$SLURM_JOB_ID"
    --slurm_array_task_id="$SLURM_ARRAY_TASK_ID"
    --wd_warmup_iters={args.wd_warmup_iters}
    --wd_anneal_iters={args.wd_anneal_iters}
    --adaptive_optimizer={args.adaptive_optimizer}
    --use_rope={args.use_rope}
    --rope_theta={args.rope_theta}
)

srun --export=ALL,MASTER_ADDR,MASTER_PORT,WORKDIR,requirements \\
    torchrun "${{DISTRIBUTED_ARGS[@]}}" {TRAINING_SCRIPT} "${{TRAINING_ARGS[@]}}" &

SRUN_PID=$!
wait $SRUN_PID
"""

try:
    with open(os.path.join(args.sbatch_logging_dir, "sbatch_command.sh"), 'w') as f:
        f.write(shell_script)
    # Use --wait so sbatch blocks until the job completes.
    # This lets the SLURM array concurrency limit control concurrent trainings.
    process = subprocess.run(['sbatch', '--wait'], input=shell_script, text=True, capture_output=True, check=True)
    print(f"Job submitted successfully: {process.stdout.strip()}", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error submitting job: {e.stderr}", flush=True)
    raise RuntimeError("Failed to submit the sbatch job. Please check the error message above.")
