# Config for varying width only
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/test_config.py --max_concurrent 30 

import numpy as np

from model_family_configs import (
    width_only,
    depth_only,
    joint_width_depth,
    head_size,
    kv_reps,
    joint_gqa,
    kv_reps_thin,
    kv_reps_small,
)

WEIGHT_DECAY = 0.0
WANDB_PROJECT = 'config-tests-kv-small-no-moe'

PROD = False

base_configs = [
    # *width_only, 
    # *depth_only, 
    # *joint_width_depth, 
    # *joint_gqa, 
    # *head_size, 
    # *kv_reps,
    *kv_reps_small,
]
configs = []

for cfg in base_configs:
    new_config = cfg.get_config(prod=PROD)
    new_config['weight_decay'] = WEIGHT_DECAY
    new_config['learning_rate'] = 1e-3
    new_config['log_wandb'] = 'true'
    new_config['wandb_project'] = WANDB_PROJECT
    new_config['qos'] = 'lowprio'
    new_config['partition'] = 'lowprio'
    new_config['reservation'] = 'moe'

    configs.append(new_config)

if __name__ == "__main__":
    import json

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))
