#!/bin/bash
for dir in $(cat directories.txt); do mkdir hparams/$dir ; cp hparams/HalfCheetah-v4/asac.yaml hparams/"$dir"/asac.yaml; done