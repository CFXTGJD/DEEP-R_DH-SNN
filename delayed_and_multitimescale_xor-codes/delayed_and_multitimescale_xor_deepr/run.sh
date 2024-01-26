#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=sbatch_example
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time 12:00:00

wandb offline
python delayed_xor_snn_deepr.py --branch 2 --use_even_mask=True
python delayed_xor_snn_deepr.py --branch 2 --use_even_mask=False
python multi_xor_snn_deepr.py --branch 2 --use_even_mask=True
python multi_xor_snn_deepr.py --branch 2 --use_even_mask=False