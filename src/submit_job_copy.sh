#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --job-name=AugustPython
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=output.log
#SBATCH --error=error.log

# Load any necessary modules
module load anaconda3/2022.05

# Change to the directory where your script is located
cd /work/sds-lab/august/crops/src

# Run the Python script
python 	ml-4A-use-all-data-PY.py
