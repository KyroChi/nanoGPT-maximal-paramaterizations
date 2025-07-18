#!/bin/bash

#SBATCH --job-name=nanogpt-2node    # Job name
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks-per-node=1         # One task (torchrun process) per node
#SBATCH --gres=gpu:8                # Request 8 GPUs per node (adjust as per your nodes' GPU count)
#SBATCH --cpus-per-task=32          # Number of CPU cores per task (adjust based on your setup and needs)
#SBATCH --mem=256G                  # Memory per node (adjust as needed)
#SBATCH --time=24:00:00             # Wall clock time limit (HH:MM:SS)
#SBATCH --output=slurm_output/%x-%j.out  # Standard output and error log
#SBATCH --error=slurm_output/%x-%j.err   # Error log
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --exclusive
#SBATCH --distribution=block:block

GPUS_PER_NODE=8
NNODES=$SLURM_JOB_NUM_NODES

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export TRITON_CACHE_DIR="/tmp/triton-cache"

echo Node IP: $head_node_ip
echo $SLURM_JOB_NODELIST
export LOGLEVEL=INFO

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --rdzv_id $RANDOM-$USER
  --rdzv_backend c10d
  --rdzv_endpoint $head_node_ip:29500
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} train.py config/train_shakespeare_char.py --compile=False"
srun --export=ALL,MASTER_ADDR,MASTER_PORT,WORKDIR,cmd,requirements \
  --container-mounts="/mnt:/mnt" \
  $cmd