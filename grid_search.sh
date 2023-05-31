#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=6GB
module load Miniconda3
source activate HWR2023
python grid_search.py --filename LeNet5_1 --epochs 80