#!/bin/bash
#SBATCH --job-name=u
#SBATCH --output=u-%A_%a.out
#SBATCH --error=u-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
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
ENVNAME=${1:-"PongNoFrameskip-v4"}
EXPNAME=${2:-"atari"}
ALGO=${3:-"asql"}
#python experiments/sweep.py --count=1 --project="u-chi-learning" --algo=$ALGO --exp-name $EXPNAME --env=$ENVNAME &
#python experiments/sweep.py --count=1 --project="u-chi-learning" --algo=$ALGO --exp-name $EXPNAME --env=$ENVNAME &
#python experiments/sweep.py --count=1 --project="u-chi-learning" --algo=$ALGO --exp-name $EXPNAME --env=$ENVNAME &
#python experiments/sweep.py --count=1 --project="u-chi-learning" --algo=$ALGO --exp-name $EXPNAME --env=$ENVNAME &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo=$ALGO --exp-name $EXPNAME --env=$ENVNAME &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo=$ALGO --exp-name $EXPNAME --env=$ENVNAME &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo=$ALGO --exp-name $EXPNAME --env=$ENVNAME &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo=$ALGO --exp-name $EXPNAME --env=$ENVNAME