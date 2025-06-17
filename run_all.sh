#!/bin/bash
# Dataset: dataset1
#python train_eval.py --model_type lstm --dataset_type dataset1
#python train_eval.py --model_type gru --dataset_type dataset1
#python train_eval.py --model_type transformer --dataset_type dataset1
#python train_eval.py --model_type tcn --dataset_type dataset1

# Dataset: dataset2
#python train_eval.py --model_type lstm --dataset_type dataset2
#python train_eval.py --model_type gru --dataset_type dataset2
#python train_eval.py --model_type transformer --dataset_type dataset2
#python train_eval.py --model_type tcn --dataset_type dataset2

# use video transformer
python train_eval.py --model_type video_tf --n_frames 32 --dataset_type dataset1 --learning_rate 1e-4 --batch_size 8
#python train_eval.py --model_type video_tf --n_frames 8 --dataset_type dataset1  --epochs 1 --batch_size 2

