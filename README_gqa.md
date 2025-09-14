# Running GQA + MoE Stuff

## Installation
```
    conda env create -n nanogpt -f environment.yml
    pip uninstall torch
    pip install torch
```
(please someone fix this insane hack) followed by
```
    conda activate nanogpt
```
Next, set up WandB using
```
export WANDB_API_KEY=<your_api_key>
wandb login
```
Finally, we will need to set up the OpenWebText dataset which we are using for ablations
```
    python data/openwebtext/prepare.py
```
which should only take a few minutes to download and parse.

While this is downloading, modify the files `mup_paper_experiments/build_orchastrator.py` (lines 57 and 58) and `/mnt/weka/home/kyle.chickering/code/test-share-gpt/paramaterized_train.py` (lines 44 and 45) to set the correct qos and partition parameters for the launched jobs.

To make sure that the installation is working we can submit a dry-run SLURM job to run an ablation test
```
python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/width_only.py --max_concurrent 30 --dry-run
```
Note that the SLURM logs are helpful and can be found saved in `mup_paper_experiments/slurm_logs`, saved in order of launch time.

Check:
1. That the SLURM job completed.
2. That things are being logged correctly to WandB.

## More Information About Ablation Pipeline

**muP Implementations File:** The file `mup_implementations.py` contains the defined muP implementations. Please look at the file to find out how to add new implementations.

**Config File:** These files are `mup_paper_experiments/configs/` and are simply Python scripts which define what the paramters are for the various training jobs. We can use these to define what needs to be computed during the experiment. Typically I copy and paste from a similar file and make modifications as needed. For some example ablation configs look at
- `mup_paper_experiments/configs/r_ablations_2.py`: Tests that with everything fixed except for the number of kv_reps that we get transfer.
- `mup_paper_experiments/configs/depth_only.py`: Ablation over depth. Everything is fixed and depth is varied.

## Some Ablation Guidelines
1. Adjust batch size according to Joel's batch size scaling rule.
1. Change as few things as possible.
1. Use zero weight-decay if possible.'

## Files that you hopefully don't need to touch
- `slimpj_train_2.py`: This file handles the training loop.
- `model_moe_kyle.py`: This file is where the model itself is defined.
- `paramaterized_train.py`: Wraps `slimpj_train_2.py` in a way that is convinient for calling with scripts.
- `mup_paper_experiments/build_orchestrator.py`: This takes a config file and luanches the corresponding training job.