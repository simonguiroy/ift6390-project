#!/bin/bash -l
sbatch -t 6:00:00 run.sl --dataset cifar --batch_size 64
sbatch -t 6:00:00 run.sl --dataset cifar --lr 0.0001
sbatch -t 6:00:00 run.sl --dataset cifar --lenet
sbatch -t 6:00:00 run.sl --dataset cifar --weight_decay 0.0001
sbatch -t 6:00:00 run.sl --dataset cifar --rec_coeff 0.001
sbatch -t 6:00:00 run.sl --dataset cifar --no_reconstruction
sbatch -t 6:00:00 run.sl --dataset cifar --kernel_size 5
sbatch -t 6:00:00 run.sl --dataset cifar --rout_iter 6
sbatch -t 6:00:00 run.sl --dataset cifar --primary_capsule_length 16
sbatch -t 6:00:00 run.sl --dataset cifar --weight_decay 0.00001
sbatch -t 6:00:00 run.sl --dataset cifar --batch_size 64 --no_reconstruction
