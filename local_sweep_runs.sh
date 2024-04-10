#!/bin/bash
SWEEP_ID=${1:-"none"}
ENVNAME=${2:-"LunarLander-v2"}
DEVICE=${3:-"cuda"}
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo="logu" --sweep-id=$SWEEP_ID --env_id=$ENVNAME --device=$DEVICE &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo="logu" --sweep-id=$SWEEP_ID --env_id=$ENVNAME --device=$DEVICE &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo="logu" --sweep-id=$SWEEP_ID --env_id=$ENVNAME --device=$DEVICE &
python experiments/sweep.py --count=1 --project="u-chi-learning" --algo="logu" --sweep-id=$SWEEP_ID --env_id=$ENVNAME --device=$DEVICE