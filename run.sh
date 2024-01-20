#!/bin/bash
#SBATCH --job-name=u-chi
#SBATCH --time=2-23:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --cpus-per-task=1

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=outfiles/%j.err
#SBATCH --output=outfiles/%j.out
##SBATCH --partition=AMD6276
#SBATCH --partition=Intel2650
# use the gpu:
##SBATCH --gres=gpu:1
##SBATCH --partition=DGXA100
##SBATCH --export=NONE
#SBATCH --array=1-50
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

# python experiments/wandb_job.py -d cpu -env CartPole-v1
# python experiments/wandb_job.py -d cuda -env PongNoFrameskip-v4

# python darer/LogUAgent.py
python experiments/wandb_job.py -env MountainCar-v0
# python experiments/baselines/DQN_comparison.py
# python local_finetuned_runs.py -a logu

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"