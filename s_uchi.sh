#!/bin/bash
#SBATCH --job-name=logu
#SBATCH --output=logu-%A_%a.out
#SBATCH --error=logu-%A_%a.err
#SBATCH --time=48:00:00
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
SWEEP_ID=${1:-"none"}
ENVNAME=${2:-"LunarLander-v2"}
DEVICE=${3:-"cuda"}
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo="logu" --sweep-id=$SWEEP_ID --env_id=$ENVNAME --device=$DEVICE &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo="logu" --sweep-id=$SWEEP_ID --env_id=$ENVNAME --device=$DEVICE &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo="logu" --sweep-id=$SWEEP_ID --env_id=$ENVNAME --device=$DEVICE &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo="logu" --sweep-id=$SWEEP_ID --env_id=$ENVNAME --device=$DEVICE
