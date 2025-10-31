# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/moe_granularity_large.py --max_concurrent 30


import numpy as np
from copy import deepcopy 

PROD = False
NO_PROD_ITERS = 100

WEIGHT_DECAY = 0.05
WANDB_PROJECT = 'moe-ablate-granularity-new-scaling-large'

HIDDEN_DIM_BASE = 256
FFN_HIDDEN_DIM_BASE = 128

MOE_SEQ_AUX_LOSS_COEFF = 0.002 / 64

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 250,
    'eval_iters': 20,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'avg_interval': 120,
    'sbatch_mem': 512,
    'dataset': 'openwebtext',
    'use_moe': True,
    'block_size': 8192,
    'moe_seq_aux_loss_coeff': MOE_SEQ_AUX_LOSS_COEFF,
    'enable_fsdp': True,
    'compile': False,
    'dtype': 'bfloat16',
    'n_embd': 1024,
    'n_head': 8,
    'n_kv_head': 8,
    'n_layer': 3,
    'num_experts': 32,
    'sbatch_timeout': '02:00:00' if PROD else '00:05:00',   
    'partition': 'lowprio',
    'qos': 'lowprio'
}

model_configs = [
    {
        'router_topk': 1,
        'moe_ffn_hidden_size': 4096,
        'n_gpus': 6,
        'gradient_accumulation_steps': 6,
        'batch_size': 5,
        'max_iters': 2814 if PROD else NO_PROD_ITERS,
    },
    {
        'router_topk': 2,
        'moe_ffn_hidden_size': 2048,
        'n_gpus': 5,
        'gradient_accumulation_steps': 5,
        'batch_size': 5,
        'max_iters': 1902 if PROD else NO_PROD_ITERS,
    },
    {
        'router_topk': 4,
        'moe_ffn_hidden_size': 1024,
        'n_gpus': 4,
        'gradient_accumulation_steps': 4,
        'batch_size': 5,
        'max_iters': 1454 if PROD else NO_PROD_ITERS,
    },
    {
        'router_topk': 8,
        'moe_ffn_hidden_size': 512,
        'n_gpus': 3,
        'gradient_accumulation_steps': 3,
        'batch_size': 5,
        'max_iters': 1324 if PROD else NO_PROD_ITERS,
    },
    {
        'router_topk': 16,
        'moe_ffn_hidden_size': 256,
        'n_gpus': 3,
        'gradient_accumulation_steps': 3,
        'batch_size': 5,
        'max_iters': 1016 if PROD else NO_PROD_ITERS,
    },
]

LEARNING_RATE_SAMPLES = 13
learning_rates = [
    10**p for p in np.linspace(-3.1, -2.2, LEARNING_RATE_SAMPLES)
] if PROD else [0.002512]
seeds = [42, 43, 44] if PROD else [42]

configs = []
for seed in seeds:
    for conf in model_configs:
        for lr in learning_rates:
            for mup in [True]: #, False]:
                conf = deepcopy(conf)
                conf.update(shared_params)

                # if not mup:
                #     lr = lr / (conf['n_embd'] / HIDDEN_DIM_BASE)

                conf['learning_rate'] = lr
                conf['warmup_iters'] = int(0.07 * conf['max_iters'])
                conf['seed'] = seed
                conf['min_lr'] = lr / 10
                conf['mup'] = 'true' if mup else 'false'
                conf['mup_multiplier'] = conf['n_embd'] / HIDDEN_DIM_BASE if mup else 1
                conf['moe_ffn_mup_multiplier'] = conf['moe_ffn_hidden_size'] / FFN_HIDDEN_DIM_BASE if mup else 1
                conf['impl'] = 'moe_fsdp' if mup else 'standard_param_impl'

                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))