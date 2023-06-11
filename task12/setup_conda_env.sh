#!/bin/bash
#SBATCH --job-name=testingSSDLite
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mem=8GB
module load Miniconda3
conda create -n HWR2023 python=3.10
source activate HWR2023
pip install -r requirements.txt
