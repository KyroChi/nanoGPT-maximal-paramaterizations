# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/gqa_transfer_2.py --max_concurrent 30

import numpy as np
from copy import deepcopy


WANDB_PROJECT = 'gqa-transfer-large'
WEIGHT_DECAY = 0.1

PROD = False

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 100,
    'eval_iters': 25,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'avg_interval': 120,
    'sbatch_mem': 512,
    'dataset': 'openwebtext',
    'block_size': 8192,
}

model_configs = [
    {
        'n_embd': 320, 
        'n_head': 10, 
        'n_kv_head': 2, 
        'n_layer': 8,
        'n_gpus': 1,
        'gradient_accumulation_steps': 1,
        'batch_size': 1,
        'max_iters': 868 if PROD else 30,
    },
    {
        'n_embd': 384,
        'n_layer': 10,
        'n_kv_head': 4,
        'n_head': 8,
        'n_gpus': 1,
        'gradient_accumulation_steps': 1,
        'batch_size': 1,
        'max_iters': 868 if PROD else 30,
    },
    {
        'n_embd': 512,
        'n_layer': 13,
        'n_kv_head': 4,
        'n_head': 8,
        'n_gpus': 1,
        'gradient_accumulation_steps': 1,
        'batch_size': 1,
        'max_iters': 868 if PROD else 30,
    },
    {
        'n_embd': 768,
        'n_layer': 18,
        'n_kv_head': 4,
        'n_head': 12,
        'n_gpus': 1,
        'gradient_accumulation_steps': 1,
        'batch_size': 1,
        'max_iters': 868 if PROD else 30,
        'enable_fsdp': 'true',
        'sbatch_nodes': 1,
    },
    {
        'n_embd': 1024,
        'n_layer': 18,
        'n_kv_head': 4,
        'n_head': 17,
        'n_gpus': 2,
        'gradient_accumulation_steps': 2,
        'batch_size': 1,
        'max_iters': 868 if PROD else 30,
        'enable_fsdp': 'true',
        'sbatch_nodes': 1,
    },
    {
        'n_embd': 1280,
        'n_layer': 20,
        'n_kv_head': 4,
        'n_head': 20,
        'n_gpus': 4,
        'gradient_accumulation_steps': 4,
        'batch_size': 1,
        'max_iters': 868 if PROD else 30,
        'enable_fsdp': 'true',
        'sbatch_nodes': 1,
    },
    {
        'n_embd': 1536,
        'n_layer': 24,
        'n_kv_head': 4,
        'n_head': 24,
        'n_gpus': 4,
        'gradient_accumulation_steps': 4,
        'batch_size': 1,
        'max_iters': 868 if PROD else 30,
        'enable_fsdp': 'true',
        'sbatch_nodes': 1,
        'sbatch_mem': 1024,
    },
    {
        'n_embd': 2048,
        'n_layer': 30,
        'n_kv_head': 4,
        'n_head': 32,
        'n_gpus': 8,
        'gradient_accumulation_steps': 8,
        'batch_size': 1,
        'max_iters': 868 if PROD else 30,
        'enable_fsdp': 'true',
        'sbatch_nodes': 2,
        'sbatch_mem': 1024,
    }
]

learning_rates = {
    320: [10**p for p in np.linspace(-3.0, -1.5, 13)] if PROD else [0.01],
    384: [10**p for p in np.linspace(-3.0, -1.5, 13)] if PROD else [0.01],
    512: [10**p for p in np.linspace(-3.0, -1.5, 9)] if PROD else [0.01],
    768: [10**p for p in np.linspace(-3.25, -1.5, 7)] if PROD else [0.01],
    1024: [10**p for p in np.linspace(-3.25, -1.5, 5)] if PROD else [0.01],
    1280: [10**p for p in np.linspace(-3.25, -1.5, 3)] if PROD else [0.01],
    1536: [10**p for p in np.linspace(-3.5, -1.5, 3)] if PROD else [0.01],
    2048: [10**p for p in np.linspace(-3.5, -1.5, 3)] if PROD else [0.01],
}

seeds_per_model = {
    320: [42, 43, 44] if PROD else [42],
    384: [42, 43, 44] if PROD else [42],
    512: [42, 43] if PROD else [42],
    768: [42, 43] if PROD else [42],
    1024: [42] if PROD else [42],
    1280: [42] if PROD else [42],
    1536: [42] if PROD else [42],
    2048: [42] if PROD else [42],
}

configs = []
for mup in [True]: 
    for conf in model_configs:
        embed_size = conf['n_embd']
        rates = learning_rates[embed_size]
        seeds = seeds_per_model[embed_size]
        for seed in seeds:
            for lr in rates:
                conf = deepcopy(conf)
                conf['learning_rate'] = lr
                conf['seed'] = seed
                conf['min_lr'] = lr / 10
                conf['mup'] = 'true' if mup else 'false'
                conf['mup_multiplier'] = conf['n_embd'] / 512 if mup else 1
                conf['impl'] = 'kyle_impl' if mup else 'standard_param_impl'

                conf.update(shared_params)
                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))