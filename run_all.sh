#!/bin/bash

python train_eval.py --model_type lstm
python train_eval.py --model_type gru
python train_eval.py --model_type transformer
python train_eval.py --model_type tcn