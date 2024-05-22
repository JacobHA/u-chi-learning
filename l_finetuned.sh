#!/bin/bash
ENVNAME=${1:-"AsterixNoFrameskip-v4"}
DEVICE=${2:-"cuda:0"}
ALGO=${3:-"asql"}
N=${3:-"0"}

python experiments/finetuned_runs.py --env_id=$ENVNAME --device=$DEVICE --count 1 --algo=$ALGO &> $ENVNAME-$ALGO-$DEVICE-$N.out &
