# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/_sweep_all.py --max_concurrent 30

# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_50m_all_sweep-2 --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_50m_all_sweep-2
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_50m_all_sweep-3 --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_50m_all_sweep-3
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_50m_all_sweep-4 --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_50m_all_sweep-4
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_50m_all_sweep-5 --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_50m_all_sweep-5

# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_50m_all_sweep-unit-wd --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_50m_all_sweep-unit-wd
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_all_sweep-sp --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_all_sweep-sp
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_all_sweep-new_kv_2 --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_all_sweep-new_kv_2

# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_jwa_all_sweep-unit-wd --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_jwa_all_sweep-unit-wd
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_jwa_all_sweep-new_kv_2 --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_jwa_all_sweep-new_kv_2
# python crawl_wandb.py --entity krchickering-uc-davis --project terminal_model_jwa_all_sweep-sp --output-dir ./mup_paper_experiments/results/sweeps/terminal_model_jwa_all_sweep-sp


import numpy as np

from model_family_configs import (
    terminal_models,
    joint_width_depth
)

WANDB_PROJECT = 'terminal_model_jwa_all_sweep-unit-wd'

PROD = True

# base_configs = [
#     terminal_models['50m'],
# ]
base_configs = joint_width_depth[:-1]
configs = []

def sample_hyperparameters(n_samples=500, seed=None):
    """
    Sample hyperparameter values for learning rate, weight decay, and std deviation.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (lr_values, wd_values, std_values) arrays of length n_samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    # lr between 10^{-3.15, -1.5}, centered around 0.0034475
    # Convert center to log space: log10(0.0034475) ≈ -2.46
    lr_log_min, lr_log_max = -3.15, -1.5
    lr_log_center = np.log10(0.0034475)
    
    # wd between 10^{-4, -0.15}, centered around 0.01
    # Convert center to log space: log10(0.01) = -2
    wd_log_min, wd_log_max = -4, -0.15
    wd_log_center = np.log10(0.01)
    
    # std between 0.001 and 0.1, centered around 0.05
    std_min, std_max = 0.001, 0.1
    std_center = 0.05
    
    # Sample in log space for lr and wd (log-uniform distribution)
    lr_log_values = np.random.uniform(lr_log_min, lr_log_max, n_samples)
    wd_log_values = np.random.uniform(wd_log_min, wd_log_max, n_samples)
    
    # Convert back to linear space
    lr_values = 10 ** lr_log_values
    wd_values = 10 ** wd_log_values
    
    # Sample std uniformly in linear space
    # std_values = np.random.uniform(std_min, std_max, n_samples)
    std_values = 0.05 * np.ones(n_samples)
    
    return lr_values, wd_values, std_values

# Sample 500 values in 
# lr between 10^{-3.15, -1.5}, centered around 0.0034475
# wd between 10^{-4, -0.15}, centered around 0.01
# std between 0.001 and 0.1, centered around 0.05
samples = 250
lr_values, wd_values, std_values = sample_hyperparameters(samples)



seeds = [42] #, 43] #, 44] if PROD else [43]

for seed in seeds:
    for cfg in base_configs:
        for (lr, wd, std_dev) in zip(lr_values, wd_values, std_values):
            lr = float(lr)
            wd = float(wd)
            std_dev = float(std_dev)

            for impl in ['tpv_left_impl_unit_wd']: #, 'xllm_impl']:
                new_config = cfg.get_config(prod=PROD)
                new_config['wandb_run_name'] = f"{new_config['wandb_run_name']}-lr_{lr:.6f}-std_{std_dev:.6f}-seed_{seed}"
                new_config['weight_decay'] = wd
                new_config['learning_rate'] = lr
                new_config['min_lr'] = lr / 10
                new_config['log_wandb'] = 'true'
                new_config['wandb_project'] = WANDB_PROJECT
                new_config['seed'] = seed
                new_config['mup'] = 'false' if impl == 'standard_param_impl' else 'true'
                new_config['decay_lr'] = 'true'
                new_config['decay_profile'] = 'cosine'
                new_config['impl'] = impl
                new_config['mup_multiplier'] = new_config['n_embd'] / 384.0
                new_config['init_std'] = std_dev
                new_config['warmup_iters'] = int(0.02 * new_config['max_iters'])

                new_config['eval_interval'] = 250
                new_config['eval_iters'] = 50
                new_config['eval_interval'] = 250

                new_config['enable_fsdp'] = 'false'
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