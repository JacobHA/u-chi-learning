#!/bin/bash
#SBATCH --job-name=logu-unor-cls-%A_%a
#SBATCH --output=logu-unor-cls-%A_%a.out
#SBATCH --error=logu-unor-cls-%A_%a.err
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
ENVNAME=${1:-"LunarLander-v2"}
ALGO=${2:-"u"}
DEVICE=${3:-"cuda"}
python experiments/local_finetuned_runs.py -c 1 -a $ALGO -e $ENVNAME -d $DEVICE
