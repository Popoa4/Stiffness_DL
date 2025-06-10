#!/bin/bash
# Dataset: dataset1
python train_eval.py --model_type lstm --dataset_type dataset1
python train_eval.py --model_type gru --dataset_type dataset1
python train_eval.py --model_type transformer --dataset_type dataset1
python train_eval.py --model_type tcn --dataset_type dataset1

# Dataset: dataset2
#python train_eval.py --model_type lstm --dataset_type dataset2
#python train_eval.py --model_type gru --dataset_type dataset2
#python train_eval.py --model_type transformer --dataset_type dataset2
#python train_eval.py --model_type tcn --dataset_type dataset2