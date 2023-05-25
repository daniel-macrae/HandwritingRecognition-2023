#!/bin/bash
#SBATCH --job-name=L5_classifier
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=6GB
module load Miniconda3
source activate HWR2023
python classifier.py --model LeNet5 --epochs 50