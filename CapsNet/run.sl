#!/bin/bash -l
#SBATCH --gres=gpu
python capsule_network.py $@
