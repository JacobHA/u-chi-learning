#!/bin/bash
#SBATCH --job-name=u-chi
#SBATCH --time=1-13:00:00
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=6

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=outfiles/%j.err
#SBATCH --output=outfiles/%j.out
##SBATCH --partition=AMD6276
# use the gpu:
##SBATCH --gres=gpu:1
##SBATCH --partition=DGXA100
##SBATCH --export=NONE
#SBATCH --array=1-5
echo "using scavenger"

# Prepare conda:
eval "$(conda shell.bash hook)"
conda activate /home/jacob.adamczyk001/miniconda3/envs/oblenv
export CPATH=$CPATH:$CONDA_PREFIX/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export MUJOCO_GL="glfw"

# # Set the Weights and Biases environment variables
# export WANDB_MODE=offline
# wandb offline

# Start the evaluations
EXPNAME=${1:-"atari-pong"}
python experiments/atari_sweep.py --local-wandb False --device cpu --proj u-chi-learning --exp-name $EXPNAME --n_runs 100
