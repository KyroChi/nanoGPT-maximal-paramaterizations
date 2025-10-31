# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/small_gqa_reps.py --max_concurrent 10
# python crawl_wandb.py --entity krchickering-uc-davis --project small-gqa-reps-mup --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/small_gqa_reps-mup

import numpy as np
from model_family_configs import ModelConfig

WEIGHT_DECAY = 0.1
WANDB_PROJECT = 'small-gqa-reps-mup'
BASE_EPS = 1e-10

PROD = True

learning_rate_samples = 9 if PROD else 1
learning_rates = [2**p for p in np.linspace(-12, -2, learning_rate_samples)] if PROD else [1e-3]

base_configs = [
    ModelConfig(
        wandb_run_name='small_gqa_rep',
        n_embd=1536,
        n_layer=1,
        n_head=12,
        prod_sbatch_nodes=1,
        prod_n_gpus=4,
        prod_gradient_accumulation_steps=8,
        prod_batch_size=6,
        prod_max_iters=2060,
        prod_sbatch_mem=256,
        enable_fsdp='true',
        n_kv_head=12,
    ) 
]

reps = [1, 2, 3, 4, 6, 12]
seeds = [42]


configs = []

for seed in seeds:
    for cfg in base_configs:
        # for impl in ['tpv_left_impl_new_kv_2']:
        for impl in ['tpv_left_impl_no_kv', 'standard_param_impl']:
            mup = 'false' if impl == 'standard_param_impl' else 'true'
            for r in reps:
                for lr in learning_rates:
                    new_config = cfg.get_config(prod=PROD)
                    new_config['wandb_run_name'] = f"{new_config['wandb_run_name']}-lr_{lr}-wd_{WEIGHT_DECAY}-seed_{seed}"
                    new_config['weight_decay'] = WEIGHT_DECAY
                    new_config['learning_rate'] = lr
                    new_config['min_lr'] = lr / 10
                    new_config['log_wandb'] = 'true'
                    new_config['wandb_project'] = WANDB_PROJECT
                    new_config['seed'] = seed
                    new_config['mup'] = mup
                    new_config['decay_lr'] = 'true'
                    new_config['decay_profile'] = 'cosine'
                    new_config['impl'] = impl
                    new_config['eps'] = BASE_EPS  / new_config['n_embd'] if mup == 'true' else BASE_EPS
                    new_config['n_kv_head'] = new_config['n_head'] // r

                    new_config['mup_multiplier'] = 1.0 

                    new_config['avg_interval'] = 120
                    new_config['eval_interval'] = 500
                    new_config['eval_iters'] = 250

                    new_config['enable_fsdp'] = 'false'
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
