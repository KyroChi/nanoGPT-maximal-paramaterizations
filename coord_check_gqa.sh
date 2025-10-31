#!/bin/bash
# filepath: /mnt/weka/home/kyle.chickering/code/nanoGPT/coord_check_gqa.sh
#SBATCH --time=5:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-6
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --distribution=pack

EXPS=(
    "48 576 12 1"
    "48 576 12 2"
    "48 576 12 3"
    "48 576 12 4"
    "48 576 12 6"
    "48 576 12 12"
)

# Calculate the index into the EXPS array
index=$SLURM_ARRAY_TASK_ID

# Check if the index is within the bounds of the array
if [ $index -ge 0 ] && [ $index -lt ${#EXPS[@]} ]; then
    item="${EXPS[$index]}"

    head_size=$(echo $item | cut -d' ' -f1)
    model_size=$(echo $item | cut -d' ' -f2)
    n_heads=$(echo $item | cut -d' ' -f3)
    n_kv_heads=$(echo $item | cut -d' ' -f4)
    
    echo "head_size: ${head_size}, model_size: ${model_size}, n_heads: ${n_heads}, n_kv_heads: ${n_kv_heads}"
    
    now=$(date +%Y-%m-%d_%H-%M-%S)
    out_dir=coord-check-impl/${SLURM_ARRAY_JOB_ID}
    mkdir -p ${out_dir}

    for seed in {0..10}; 
    do
        mup_multiplier=$(echo "scale=2; $model_size / 256" | bc)

        srun python coord_check.py \
            --out_dir=${out_dir} \
            --eval_interval=10000000 \
            --log_interval=10000000 \
            --eval_iters=1 \
            --eval_only=False \
            --init_from='scratch' \
            --wandb_log=False \
            --dataset='openwebtext' \
            --gradient_accumulation_steps=1 \
            --batch_size=1 \
            --block_size=1024 \
            --n_layer=8 \
            --n_head=${n_heads} \
            --n_kv_head=${n_kv_heads} \
            --n_embd=${model_size} \
            --dropout=0.0 \
            --bias=False \
            --init_std=0.02 \
            --learning_rate=4e-5 \
            --max_iters=4 \
            --eps=1e-10 \
            --weight_decay=0.0 \
            --beta1=0.9 \
            --beta2=0.95 \
            --grad_clip=1.0 \
            --decay_lr=False \
            --mup=True \
            --mup_multiplier=${mup_multiplier} \
            --seed=${seed} \
            --backend='nccl' \
            --device='cuda' \
            --dtype='float32' \
            --compile=False \
            --impl='tpv_left_impl_no_kv' \
            --tag="${item}"
    done
else
    echo "SLURM_ARRAY_TASK_ID is out of range (0-${#EXPS[@]})."
fi

# #!/bin/bash
# #SBATCH --time=5:30:00
# #SBATCH --nodes=1
# #SBATCH --exclusive
# #SBATCH --gres=gpu:1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4

# # EXPS=(
# #     "16 256 16 2"
# #     "16 256 16 4"
# #     "16 256 16 8"
# #     "16 384 24 2"
# #     "16 384 24 3"
# #     "16 384 24 4"
# #     "16 384 24 6"
# #     "16 384 24 8"
# #     "16 384 24 12"
# #     "16 512 32 2"
# #     "16 512 32 4"
# #     "16 512 32 8"
# #     "16 512 32 16"
# #     "16 640 40 2"
# #     "16 640 40 4"
# #     "16 640 40 5"
# #     "16 640 40 8"
# #     "16 640 40 10"
# #     "16 640 40 20"
# #     "16 768 48 2"
# #     "16 768 48 3"
# #     "16 768 48 4"
# #     "16 768 48 6"
# #     "16 768 48 8"
# #     "16 768 48 12"
# #     "16 768 48 16"
# #     "16 768 48 24"
# #     "16 896 56 2"
# #     "16 896 56 4"
# #     "16 896 56 7"
# #     "16 896 56 8"
# #     "16 896 56 14"
# #     "16 896 56 28"
# #     "16 1024 64 2"
# #     "16 1024 64 4"
# #     "16 1024 64 8"
# #     "16 1024 64 16"
# #     "16 1024 64 32"
# #     "16 1152 72 2"
# #     "16 1152 72 3"
# #     "16 1152 72 4"
# #     "16 1152 72 6"
# #     "16 1152 72 8"
# #     "16 1152 72 9"
# #     "16 1152 72 12"
# #     "16 1152 72 18"
# #     "16 1152 72 24"
# #     "16 1152 72 36"
# #     "16 1280 80 2"
# #     "16 1280 80 4"
# #     "16 1280 80 5"
# #     "16 1280 80 8"
# #     "16 1280 80 10"
# #     "16 1280 80 16"
# #     "16 1280 80 20"
# #     "16 1280 80 40"
# #     "32 256 8 2"
# #     "32 256 8 4"
# #     "32 384 12 2"
# #     "32 384 12 3"
# #     "32 384 12 4"
# #     "32 384 12 6"
# #     "32 512 16 2"
# #     "32 512 16 4"
# #     "32 512 16 8"
# #     "32 640 20 2"
# #     "32 640 20 4"
# #     "32 640 20 5"
# #     "32 640 20 10"
# #     "32 768 24 2"
# #     "32 768 24 3"
# #     "32 768 24 4"
# #     "32 768 24 6"
# #     "32 768 24 8"
# #     "32 768 24 12"
# #     "32 896 28 2"
# #     "32 896 28 4"
# #     "32 896 28 7"
# #     "32 896 28 14"
# #     "32 1024 32 2"
# #     "32 1024 32 4"
# #     "32 1024 32 8"
# #     "32 1024 32 16"
# #     "32 1152 36 2"
# #     "32 1152 36 3"
# #     "32 1152 36 4"
# #     "32 1152 36 6"
# #     "32 1152 36 9"
# #     "32 1152 36 12"
# #     "32 1152 36 18"
# #     "32 1280 40 2"
# #     "32 1280 40 4"
# #     "32 1280 40 5"
# #     "32 1280 40 8"
# #     "32 1280 40 10"
# #     "32 1280 40 20"
# #     "48 384 8 2"
# #     "48 384 8 4"
# #     "48 768 16 2"
# #     "48 768 16 4"
# #     "48 768 16 8"
# #     "48 1152 24 2"
# #     "48 1152 24 3"
# #     "48 1152 24 4"
# #     "48 1152 24 6"
# #     "48 1152 24 8"
# #     "48 1152 24 12"
# #     "64 256 4 2"
# #     "64 384 6 2"
# #     "64 384 6 3"
# #     "64 512 8 2"
# #     "64 512 8 4"
# #     "64 640 10 2"
# #     "64 640 10 5"
# #     "64 768 12 2"
# #     "64 768 12 3"
# #     "64 768 12 4"
# #     "64 768 12 6"
# #     "64 896 14 2"
# #     "64 896 14 7"
# #     "64 1024 16 2"
# #     "64 1024 16 4"
# #     "64 1024 16 8"
# #     "64 1152 18 2"
# #     "64 1152 18 3"
# #     "64 1152 18 6"
# #     "64 1152 18 9"
# #     "64 1280 20 2"
# #     "64 1280 20 4"
# #     "64 1280 20 5"
# #     "64 1280 20 10"
# #     "80 640 8 2"
# #     "80 640 8 4"
# #     "80 1280 16 2"
# #     "80 1280 16 4"
# #     "80 1280 16 8"
# #     "96 384 4 2"
# #     "96 768 8 2"
# #     "96 768 8 4"
# #     "96 1152 12 2"
# #     "96 1152 12 3"
# #     "96 1152 12 4"
# #     "96 1152 12 6"
# #     "112 896 8 2"
# #     "112 896 8 4"
# #     "128 512 4 2"
# #     "128 768 6 2"
# #     "128 768 6 3"
# #     "128 1024 8 2"
# #     "128 1024 8 4"
# #     "128 1152 9 3"
# #     "128 1280 10 2"
# #     "128 1280 10 5"
# # )

# S=(
#     "16 256 16 2"
#     "16 256 16 4"
#     "16 256 16 8"
#     "16 384 24 2"
#     "16 384 24 3"
#     "16 384 24 4"
#     "16 384 24 6"
#     "16 384 24 8"
#     "16 384 24 12"
#     "16 512 32 2"
#     "16 512 32 4"
#     "16 512 32 8"
#     "16 512 32 16"
#     "16 640 40 2"
#     "16 640 40 4"
#     "16 640 40 5"
#     "16 640 40 8"
# )

# now=$(date +%Y-%m-%d_%H-%M-%S)
# out_dir=coord-check-impl/${now}
# mkdir -p ${out_dir}

# for item in "${EXPS[@]}"; 
# do
#     head_size=$(echo $item | cut -d' ' -f1)
#     model_size=$(echo $item | cut -d' ' -f2)
#     n_heads=$(echo $item | cut -d' ' -f3)
#     n_kv_heads=$(echo $item | cut -d' ' -f4)
    
#     echo "head_size: ${head_size}, model_size: ${model_size}, n_heads: ${n_heads}, n_kv_heads: ${n_kv_heads}"
#     for seed in {0..5}; 
#     do
#         mup_multiplier=$(echo "scale=2; $model_size / 256" | bc)

#         srun python coord_check.py \
#             --out_dir=${out_dir} \
#             --eval_interval=10000000 \
#             --log_interval=10000000 \
#             --eval_iters=1 \
#             --eval_only=False \
#             --init_from='scratch' \
#             --wandb_log=False \
#             --dataset='slim_pajama' \
#             --gradient_accumulation_steps=1 \
#             --batch_size=1 \
#             --block_size=1024 \
#             --n_layer=2 \
#             --n_head=${n_heads} \
#             --n_kv_head=${n_kv_heads} \
#             --n_embd=${model_size} \
#             --dropout=0.0 \
#             --bias=False \
#             --init_std=0.02 \
#             --learning_rate=4e-5 \
#             --max_iters=4 \
#             --weight_decay=0.0 \
#             --beta1=0.9 \
#             --beta2=0.95 \
#             --grad_clip=1.0 \
#             --decay_lr=False \
#             --mup=True \
#             --mup_multiplier=${mup_multiplier} \
#             --seed=${seed} \
#             --backend='nccl' \
#             --device='cuda' \
#             --dtype='float32' \
#             --compile=False \
#             --impl='tpv_left_impl' \
#             --tag=${item}
#     done
# done