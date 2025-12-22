# Config for 124M active parameter MoE model
# Architecture: GPT-2 style with MoE replacing MLP
# - 6 experts per layer, top-2 routing
# - Active params ~124M, total params ~238M

import os

# wandb logging
wandb_log = True
wandb_project = 'nanogpt-moe'
wandb_run_name = 'gpt2-moe-124m-kyro'

# output directory
out_dir = f'out/{wandb_run_name}'
os.makedirs(out_dir, exist_ok=True)

# model architecture (matches GPT-2 124M for attention/embeddings)
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
vocab_size = 50304  # padded for efficiency
bias = False  # no bias for slightly better performance
dropout = 0.0

# MoE configuration
use_moe = True
# Option 1: Uniform configuration (same for all layers)
# moe_num_experts = 6
# moe_num_experts_per_tok = 2  # top-2 routing

# Option 2: Per-layer configuration (uncomment to use)
# Specify different number of experts per layer (must have n_layer elements)
moe_num_experts = [3, 4, 5, 7, 8, 9, 9, 8, 7, 5, 4, 3]  # More experts in early/late layers
# Specify different top-k per layer (must have n_layer elements)
moe_num_experts_per_tok = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # Top-1 in middle layers

# Hidden size chosen so active params ≈ 124M
# Each expert: 768 * 1536 * 2 = 2.36M params
# Top-2 active: 4.72M per layer (matches GPT-2 MLP)
moe_ffn_hidden_size = 1536

# training configuration
batch_size = 12  # micro batch size per GPU
block_size = 1024
gradient_accumulation_steps = 40  # effective batch = 12 * 40 * num_gpus

# optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# evaluation
eval_interval = 1000
eval_iters = 200
log_interval = 10

# checkpointing
always_save_checkpoint = False

# system
compile = False  # disable for FSDP compatibility

