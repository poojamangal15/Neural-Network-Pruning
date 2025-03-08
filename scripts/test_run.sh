#!/bin/bash

#SBATCH --job-name=resNet                      # Job name
#SBATCH --output=scripts/resNet_%j.out        # Standard output log
#SBATCH --error=scripts/resNet_%j.err         # Standard error log
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=8                        # Number of CPU cores per task
#SBATCH --gres=gpu:a100:1                        # Request one A100 GPU
#SBATCH --mem=120G                               # Memory pool
#SBATCH --time=0-04:00:00                           # 2 days, 0 hours, 0 minutes, 0 seconds
#SBATCH --partition=gpu                          # GPU partition to submit to

# Load modules
module load 2023
module load CUDA/12.1.1                          # Load necessary CUDA module
module load Python/3.11.3-GCCcore-12.3.0         # Load necessary Python module

# Activate virtual environment (if using one)
source venv/bin/activate

# Change to job submission directory
cd $SLURM_SUBMIT_DIR

# Run your script
python high_level_pruner.py


#SBATCH --time=2-00:00:00   # D-HH:MM:SS
