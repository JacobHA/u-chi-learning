#!/bin/bash
#SBATCH --job-name=bb-tria-%A_%a
#SBATCH --output=bb-tria-%A_%a.out
#SBATCH --error=bb-tria-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Load the required modules
module load anaconda/3.9
# activate conda
source /home/$USER/.bashrc
conda activate myenv

# Set the Weights and Biases environment variables
export WANDB_MODE=offline
wandb offline

# Start the evaluations
EXPNAME=${1:-"atari-v0"}
python atari_sweep.py --local-wandb True --proj u-chi-learning --exp-name $EXPNAME
