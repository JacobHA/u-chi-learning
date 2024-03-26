#!/bin/bash
#SBATCH --job-name=logu-sql-cls-%A_%a
#SBATCH --output=logu-sql-cls-%A_%a.out
#SBATCH --error=logu-sql-cls-%A_%a.err
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
DEVICE=${2:-"cuda"}
python experiments/classic_sweep.py --n_runs=1 --proj=u-chi-learning --algo="sql" --exp-name="classic-bench-v2" --env_id=$ENVNAME --device=$DEVICE &
python experiments/classic_sweep.py --n_runs=1 --proj=u-chi-learning --algo="sql" --exp-name="classic-bench-v2" --env_id=$ENVNAME --device=$DEVICE &
python experiments/classic_sweep.py --n_runs=1 --proj=u-chi-learning --algo="sql" --exp-name="classic-bench-v2" --env_id=$ENVNAME --device=$DEVICE &
python experiments/classic_sweep.py --n_runs=1 --proj=u-chi-learning --algo="sql" --exp-name="classic-bench-v2" --env_id=$ENVNAME --device=$DEVICE

