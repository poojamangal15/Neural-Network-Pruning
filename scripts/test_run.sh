#!/bin/bash

#SBATCH --job-name=depGraph                      # Job name
#SBATCH --output=scripts/dep_graph_%j.out        # Standard output log
#SBATCH --error=scripts/dep_graph_%j.err         # Standard error log
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=8                        # Number of CPU cores per task
#SBATCH --gres=gpu:a100:1                        # Request one A100 GPU
#SBATCH --mem=120G                               # Memory pool
#SBATCH --time=20:00:00                          # 20 hours 
#SBATCH --partition=gpu_a100                          # GPU partition to submit to

# Load modules
module load 2023
module load CUDA/12.1.1                          # Load necessary CUDA module
module load Python/3.11.3-GCCcore-12.3.0         # Load necessary Python module

# Activate virtual environment (if using one)
source venv/bin/activate

# Change to job submission directory
cd $SLURM_SUBMIT_DIR

# Run your script
# python train.py
python high_level_pruner.py
# python depGraph_pruning.py
# python softPruning.py