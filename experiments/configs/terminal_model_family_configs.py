terminal_models = {
    '50m': ModelConfig(
        wandb_run_name='hkm_50m',
        n_embd=512,
        n_layer=12,
        n_head=4,
        n_kv_head=4,
        prod_sbatch_nodes=1,
        prod_n_gpus=8,
        prod_gradient_accumulation_steps=8,
        prod_batch_size=5,
        prod_max_iters=7443,
        prod_sbatch_mem=256,
        enable_fsdp='true',
    ),
}