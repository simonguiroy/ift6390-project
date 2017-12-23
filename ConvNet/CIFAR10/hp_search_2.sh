#!/bin/bash

python main.py --dropout 0.25 --epochs 200 --optim "SGD" --model "LeNetDropoutXavier" > log/lenet_dropout_025_crop_flips_xavier_sgd
python main.py --dropout 0.25 --epochs 200 --optim "Adam" --model "LeNetDropoutXavier" > log/lenet_dropout_025_crop_flips_xavier_adam
python main.py --dropout 0.1 --epochs 200 --optim "Adam" --model "LeNetDropoutXavier" > log/lenet_dropout_01_crop_flips_xavier_adam
python main.py --dropout 0.75 --epochs 200 --optim "Adam" --model "LeNetDropoutXavier" > log/lenet_dropout_075_crop_flips_xavier_adam
python main.py --dropout 0.5 --epochs 200 --optim "Adam" --model "LeNetDropoutXavier" > log/lenet_dropout_05_crop_flips_xavier_adam
