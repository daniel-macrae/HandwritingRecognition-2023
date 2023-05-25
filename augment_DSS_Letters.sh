#!/bin/bash
#SBATCH --job-name=testingSSDLite
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
source activate HWR2023
python prep_dssLetters_dataset.py
