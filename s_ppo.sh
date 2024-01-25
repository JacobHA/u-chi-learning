#!/bin/bash
#SBATCH --job-name=logu-bppo-%A_%a
#SBATCH --output=logu-bppo-%A_%a.out
#SBATCH --error=logu-bppo-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Load the required modules
module load anaconda/3.9
# activate conda
source /home/$USER/.bashrc
conda activate u-chi-learning

# Set the Weights and Biases environment variables
export WANDB_MODE=offline
wandb offline

# Start the evaluations
python experiments/ppo_baseline.py --n_runs=1 --proj=u-chi-learning
