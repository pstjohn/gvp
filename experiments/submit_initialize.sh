#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=robustmicrob
#SBATCH --job-name=rcsb_dataset
#SBATCH -o /scratch/pstjohn/%j.%x.out
#SBATCH -e /scratch/pstjohn/%j.%x.err
#SBATCH --partition=debug

source ~/.bashrc
conda activate torch

srun python initialize_dataset.py