# Config for varying width only
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/_std_dev_sweep.py --max_concurrent 30 

import numpy as np

from model_family_configs import (
    terminal_models
)

LEARNING_RATE = 0.0034475
WEIGHT_DECAY = 0.0077179
WANDB_PROJECT = 'terminal_model_50m_std_sweep'

PROD = True

base_configs = [
    terminal_models['50m'],
]
configs = []

std_dev_samples = 13
std_devs = np.linspace(0.001, 0.1, std_dev_samples) if PROD else [0.02]

seeds = [42, 43, 44] if PROD else [43]

for seed in seeds:
    for cfg in base_configs:
        for impl in ['kyle_impl', 'xllm_impl']:
            for std_dev in std_devs:
                new_config = cfg.get_config(prod=PROD)
                new_config['wandb_run_name'] = f"{new_config['wandb_run_name']}-lr_{LEARNING_RATE}-std_{std_dev}-seed_{seed}"
                new_config['weight_decay'] = WEIGHT_DECAY
                new_config['learning_rate'] = LEARNING_RATE
                new_config['min_lr'] = LEARNING_RATE / 10
                new_config['log_wandb'] = 'true'
                new_config['wandb_project'] = WANDB_PROJECT
                new_config['seed'] = seed
                new_config['mup'] = 'false' if impl == 'standard_param_impl' else 'true'
                new_config['decay_lr'] = 'true'
                new_config['decay_profile'] = 'cosine'
                new_config['impl'] = impl
                new_config['mup_multiplier'] = 1.0
                new_config['init_std'] = std_dev

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
