#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv
ENVNAME=${1:-"AlienNoFrameskip-v4"}
ALGO=${2:-"asql"}

python experiments/finetuned_runs.py --env_id=$ENVNAME --count 2 --algo=$ALGO &
python experiments/finetuned_runs.py --env_id=$ENVNAME --count 2 --algo=$ALGO &
python experiments/finetuned_runs.py --env_id=$ENVNAME --count 2 --algo=$ALGO &
python experiments/finetuned_runs.py --env_id=$ENVNAME --count 2 --algo=$ALGO &
python experiments/finetuned_runs.py --env_id=$ENVNAME --count 2 --algo=$ALGO &
python experiments/finetuned_runs.py --env_id=$ENVNAME --count 2 --algo=$ALGO &
python experiments/finetuned_runs.py --env_id=$ENVNAME --count 2 --algo=$ALGO &
python experiments/finetuned_runs.py --env_id=$ENVNAME --count 2 --algo=$ALGO
