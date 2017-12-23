#!/bin/bash

python main.py --dropout 0.1 --epochs 200 --model "LeNetDropout" --optim "SGD" > ./log/lenet_dropout_01_crop_flips
python main.py --dropout 0.25 --epochs 200 --model "LeNetDropout" --optim "SGD" > ./log/lenet_dropout_025_crop_flips_lr
python main.py --dropout 0.5 --epochs 200 --model "LeNetDropout" --optim "SGD" > ./log/lenet_dropout_05_crop_flips
python main.py --dropout 0.75 --epochs 200 --model "LeNetDropout" --optim "SGD" > ./log/lenet_dropout_75_crop_flips
python main.py --dropout 0.25 --lr 0.025 --epochs 200 --model "LeNetDropout" --optim "SGD" > ./log/lenet_dropout_025_crop_flips_lr_025
python main.py --dropout 0.25 --lr 0.05 --epochs 200 --model "LeNetDropout" --optim "SGD" > ./log/lenet_dropout_025_crop_flips_lr_05
python main.py --dropout 0.25 --lr 0.075 --epochs 200 --model "LeNetDropout" --optim "SGD" > ./log/lenet_dropout_025_crop_flips_lr_075
