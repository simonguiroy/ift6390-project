#!/bin/bash

python main_with_valid.py  --dropout 0.25 --optim "Adam" --model "LeNetDropoutXavier" --epochs 3 > ./log2/testlog1
python main_with_valid.py  --dropout 0.5 --optim "Adam" --model "LeNetDropoutXavier" --epochs 3 > ./log2/testlog2
