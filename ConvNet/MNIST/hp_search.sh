#!/bin/bash

python main.py --lr 0.01 --momentum 0.25 --epochs 30 > ./log/lenet_lr_0.01_mom_025
python main.py --lr 0.01 --epochs 30 --momentum 0.5 > ./log/lenet_lr_0.01_mom_05
python main.py --lr 0.01 --epochs 30 --momentum 0.75 > ./log/lenet_lr_0.01_mom_075
python main.py --lr 0.01 --epochs 30 --momentum 0.9 > ./log/lenet_lr_0.01_mom_09
