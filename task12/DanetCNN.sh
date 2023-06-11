#!/bin/bash
#SBATCH --job-name=DanCNN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4GB

module load Python/3.10.4-GCCcore-11.3.0
 
source /home1/s5378176/.envs/HR_env/bin/activate
 

python classifier.py --model DanNet1 --epochs 100 --filename DanNet1CNN_model

deactivate