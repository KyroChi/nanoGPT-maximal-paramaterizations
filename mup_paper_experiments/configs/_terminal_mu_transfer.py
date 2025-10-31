# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/_terminal_mu_transfer.py --max_concurrent 30 
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_mu_transfer-prod --output-dir ./mup_paper_experiments/results/ablations/terminal_mu_transfer-prod

# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_mu_transfer-4 --output-dir ./mup_paper_experiments/results/ablations/terminal_mu_transfer-4
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_mu_transfer-5 --output-dir ./mup_paper_experiments/results/ablations/terminal_mu_transfer-5

import numpy as np

from model_family_configs import (
    terminal_models,
)

# optimal_lr = 0.0034475
# optimal_wd = 0.0077179
# optimal_std = 0.0505
optimal_lr = 6.85e-03
optimal_wd = 1.12e-01
optimal_std = 5.21e-02

base_eps = 1e-9

PROD = True
WANDB_PROJECT = 'terminal_mu_transfer-6'

learning_rate_samples = 13 

full_learning_rate_sweep = [
    10**p for p in np.linspace(-3.15, -1.5, learning_rate_samples)
] 

learning_rates = {
    '50m': full_learning_rate_sweep, # 13 samples around 0.0034475
    '100m': full_learning_rate_sweep[1:10], # 9 samples around 0.0034475
    '250m': full_learning_rate_sweep[2:9], # 7 samples around 0.0034475
    '500m': full_learning_rate_sweep[3:8], # 5 samples around 0.0034475
    '1b': full_learning_rate_sweep[4:7], # 3 samples around 0.0034475
    '3b': full_learning_rate_sweep[4:7], # 3 samples around 0.0034475
}

seeds = {
    '50m': [42], #, 43, 44] if PROD else [42],
    '100m': [42], # 43, 44] if PROD else [42],
    '250m': [42], # 43] if PROD else [42],
    '500m': [42], # 43] if PROD else [42],
    '1b': [42] if PROD else [42],
    '3b': [42] if PROD else [42],
}

configs = []

for p, cfg in terminal_models.items():
    if not (p == '50m' or p == '100m' or p == '250m' or p == '500m'):
        continue
    for seed in seeds[p]:
        for lr in learning_rates[p]:
            new_config = cfg.get_config(prod=PROD)
            new_config['wandb_run_name'] = f"{new_config['wandb_run_name']}-lr_{lr}-wd_{optimal_wd}-std_{optimal_std}-seed_{seed}"
            new_config['weight_decay'] = optimal_wd
            new_config['learning_rate'] = lr
            new_config['min_lr'] = lr / 10
            new_config['log_wandb'] = 'true'
            new_config['wandb_project'] = WANDB_PROJECT
            new_config['seed'] = seed
            new_config['mup'] = 'true'
            new_config['decay_lr'] = 'true'
            new_config['decay_profile'] = 'cosine'
            new_config['impl'] = 'kyle_impl'
            new_config['mup_multiplier'] = cfg.n_embd / 512.0
            new_config['init_std'] = optimal_std
            new_config['eps'] = base_eps  / new_config['n_embd']
            new_config['warmup_iters'] = int(0.02 * new_config['max_iters'])

            new_config['avg_interval'] = 250
            new_config['dataset'] = 'openwebtext'
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