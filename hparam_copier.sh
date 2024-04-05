#!/bin/bash
for dir in $(cat directories.txt); do mkdir hparams/$dir ; cp hparams/HalfCheetah-v4/arSAC.yaml hparams/"$dir"/arSAC.yaml; done