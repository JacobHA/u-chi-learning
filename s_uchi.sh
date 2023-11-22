#!/bin/bash
#SBATCH --job-name=logu-%A_%a
#SBATCH --output=logu-%A_%a.out
#SBATCH --error=logu-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
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
EXPNAME=${1:-"atari-mini"}
python experiments/atari_sweep.py --local-wandb True --proj u-chi-learning --exp-name $EXPNAME
