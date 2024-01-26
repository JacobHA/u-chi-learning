#!/bin/bash
#SBATCH --job-name=u-chi
#SBATCH --time=2-23:00:00
#SBATCH --mem-per-cpu=32gb
#SBATCH --cpus-per-task=3

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=outfiles/%j.err
#SBATCH --output=outfiles/%j.out
##SBATCH --partition=AMD6276
##SBATCH --partition=Intel6326
##SBATCH --partition=AMD6128
# use the gpu:
#SBATCH --gres=gpu:1
#SBATCH --partition=DGXA100
#SBATCH --export=NONE
#SBATCH --array=1-4
## --begin=now+1min
echo "using scavenger"

# Prepare conda:
eval "$(conda shell.bash hook)"
conda activate /home/jacob.adamczyk001/miniconda3/envs/oblenv
export CPATH=$CPATH:$CONDA_PREFIX/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export MUJOCO_GL="glfw"


echo "Start Run"
echo `date`

# python experiments/wandb_job.py -d cpu -env LunarLander-v2 -a sql
python experiments/wandb_job.py -d cuda -env PongNoFrameskip-v4 -a u

# python darer/LogUAgent.py
# python experiments/wandb_job.py -env CartPole-v1 -a u
# python experiments/wandb_job.py -env LunarLander-v2 -a sql

# python experiments/baselines/DQN_comparison.py
# python experiments/local_finetuned_runs.py -a u

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"