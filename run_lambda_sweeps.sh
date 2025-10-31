widths=(256 384 512 640 768 896 1024)
batch_sizes=(50 67 55 70 60 58 64)
iters=(659 874 1092 1340 1582 1872 2140)
grad_accum_steps=(2 2 3 3 4 5 5)
depths=(12 12 12 14 14 16 16)
seeds=(42 43 44)
# seeds=(43)

for i in "${!widths[@]}"; 
do
    width=${widths[$i]}
    batch_size=${batch_sizes[$i]}
    grad_accum_step=${grad_accum_steps[$i]}
    depth=${depths[$i]}
    iter=${iters[$i]}
    for seed in "${seeds[@]}"; 
    do
        sbatch sweep_eta_lambda_2.sh \
            $width \
            'tpv_left_impl' \
            'everything-mup-cosine' \
            $seed \
            $batch_size \
            $grad_accum_step \
            $depth \
            $iter

        sbatch sweep_eta_lambda_2.sh \
            $width \
            'standard_param_impl' \
            'everything-sp-cosine' \
            $seed \
            $batch_size \
            $grad_accum_step \
            $depth \
            $iter
    done
done
