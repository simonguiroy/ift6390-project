#!/bin/bash

python main_with_valid.py --dropout 0.25 --epochs 50 --optim "Adam" --model "LeNetDropoutXavier" > log2/lenet_dropout_025_crop_flips_xavier_adam
python main_with_valid.py --dropout 0.1 --epochs 50 --optim "Adam" --model "LeNetDropoutXavier" > log2/lenet_dropout_01_crop_flips_xavier_adam
python main_with_valid.py --dropout 0.75 --epochs 50 --optim "Adam" --model "LeNetDropoutXavier" > log2/lenet_dropout_075_crop_flips_xavier_adam
python main_with_valid.py --dropout 0.5 --epochs 50 --optim "Adam" --model "LeNetDropoutXavier" > log2/lenet_dropout_05_crop_flips_xavier_adam
