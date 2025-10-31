# Config for varying width only
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/_learning_rate_sweep.py --max_concurrent 30 

import numpy as np

from model_family_configs import (
    terminal_models
)

WEIGHT_DECAY = 0.1
WANDB_PROJECT = 'terminal_model_50m_lr_sweep'

PROD = True

base_configs = [
    terminal_models['50m'],
]
configs = []

learning_rate_samples = 13
learning_rates_sp = [
    10**p for p in np.linspace(-3.8, -2.25, learning_rate_samples)
] if PROD else [1e-4]
learning_rates_mup = [
    10**p for p in np.linspace(-3.15, -1.5, learning_rate_samples)
] if PROD else [1e-4]

seeds = [42, 43, 44] if PROD else [43]

for seed in seeds:
    for cfg in base_configs:
        for impl in ['kyle_impl', 'standard_param_impl', 'xllm_impl']:
            for lr in learning_rates_sp if impl == 'standard_param_impl' else learning_rates_mup:
                new_config = cfg.get_config(prod=PROD)
                new_config['wandb_run_name'] = f"{new_config['wandb_run_name']}-lr_{lr}-wd_{WEIGHT_DECAY}-seed_{seed}"
                new_config['weight_decay'] = WEIGHT_DECAY
                new_config['learning_rate'] = lr
                new_config['min_lr'] = lr / 10
                new_config['log_wandb'] = 'true'
                new_config['wandb_project'] = WANDB_PROJECT
                new_config['seed'] = seed
                new_config['mup'] = 'false' if impl == 'standard_param_impl' else 'true'
                new_config['decay_lr'] = 'true'
                new_config['decay_profile'] = 'cosine'
                new_config['impl'] = impl
                new_config['mup_multiplier'] = 1.0

                new_config['eval_interval'] = 250
                new_config['eval_iters'] = 50
                new_config['eval_interval'] = 250

                new_config['enable_fsdp'] = 'true'
                new_config['partition'] = 'lowprio'
                new_config['qos'] = 'lowprio'
                new_config['reservation'] = 'moe'

                configs.append(new_config)

if __name__ == "__main__":
    import json

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))
