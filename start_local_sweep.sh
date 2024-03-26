#!/bin/bash
config=${3:-"sweeps/EVAL-MountainCar-v0.yaml"}
wandb sweep --controller $config