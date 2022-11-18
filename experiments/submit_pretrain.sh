#!/bin/bash
#SBATCH --time=2-00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --account=robustmicrob
#SBATCH --job-name=train_gvp
#SBATCH --gres=gpu:2
#SBATCH -o /scratch/pstjohn/%j.%x.out
#SBATCH -e /scratch/pstjohn/%j.%x.err

source ~/.bashrc
conda activate torch

srun python pretrain_rcsd.py
