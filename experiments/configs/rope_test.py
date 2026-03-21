# Test RoPE implementation with a single model configuration
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/rope_test.py --max_concurrent 1

import numpy as np
from copy import deepcopy 

PROD = True
WANDB_PROJECT = 'rope-test-single-model'

config = {
    'n_embd': 384,
    'n_layer': 5,
    'n_head': 12,
    'n_kv_head': 12,  # No GQA for this test
    'n_gpus': 2,
    'gradient_accumulation_steps': 2,
    'batch_size': 90,
    'dtype': 'bfloat16',
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'wandb_run_name': 'rope_test_single_model',
    'eval_interval': 50,
    'eval_iters': 100,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': 0.05,
    'learning_rate': 1e-3,
    'min_lr': 1e-4,
    'avg_interval': 120,
    'sbatch_mem': 256,
    'dataset': 'openwebtext',
    'max_iters': 1000 if PROD else 30,  # Short run for testing
    'seed': 42,
    'mup': 'false',  # Use standard parameterization for simplicity
    'mup_multiplier': 1.0,
    'impl': 'standard_param_impl',
    'use_rope': 'true',  # Enable RoPE
    'rope_theta': 10000.0,  # Standard RoPE theta
    'enable_checkpointing': 'true',  # Enable checkpointing to save the model
    'normalization': 'LayerNorm',
    'q_prelayer_normalization': 'None',
    'k_prelayer_normalization': 'None',
    'compile': 'true',
    'enable_fsdp': 'false',
    'coord_check': 'false',
    'adaptive_optimizer': 'false',
    'use_moe': 'false',
    'num_experts': 0,
    'moe_ffn_hidden_size': 128,
    'router_topk': 1,
    'moe_seq_aux_loss_coeff': 0.0,
    'moe_ffn_mup_multiplier': 1.0,
    'moe_null_expert_bias': 0.0,
    'moe_random_router': 'false',
}

configs = [config]  # Single configuration

if __name__ == "__main__":
    import json
    import random

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))