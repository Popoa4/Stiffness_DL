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
#python train_eval.py --model_type video_tf --n_frames 16 --dataset_type dataset2 --learning_rate 3e-4 --batch_size 16
#python train_eval.py --model_type video_tf --n_frames 8 --dataset_type dataset1  --epochs 1 --batch_size 2


#python run_test.py \
#        --checkpoint_path "./checkpoints/video_tf_hardness1Data/video_tf_hardness1Data_epoch_280.pth" \
#        --model_type "video_tf" \
#        --dataset_type "dataset2" \
#        --n_frames 16 \
#        --batch_size 16\
#        --vit_embed_dim 128 \
#        --vit_depth 6 \
#        --vit_n_head 4 \
#        --vit_patch_size 16




#python run_test.py --checkpoint_path "./checkpoints/video_tf_hardness5/video_tf_hardness5_epoch_200.pth" --model_type "video_tf" --dataset_type "dataset2" --n_frames 16 --batch_size 32 --vit_embed_dim 128 --vit_depth 6 --vit_n_head 4 --vit_patch_size 16
#python main.py --model_type video_tf --experiment random --dataset_type dataset2 --learning_rate 3e-4 --batch_size 16 --patience 20 --min_delta 0.001

# 实验 1 ：硬度 17 留做测试
python main.py --model_type video_tf --experiment 1 --dataset_type dataset2 --learning_rate 3e-4 --batch_size 16 --patience 20 --min_delta 0.001

# 实验 2 ：CYLIND 做测试
python main.py --model_type video_tf --experiment 2 --dataset_type dataset2 --learning_rate 3e-4 --batch_size 16 --patience 30 --min_delta 0.001

# 实验 3-basic ：BASIC 做测试
python main.py --model_type video_tf --experiment 3_basic --dataset_type dataset2 --learning_rate 3e-4 --batch_size 16 --patience 30 --min_delta 0.001

# 实验 3-choco ：CHOCO 做测试
python main.py --model_type video_tf --experiment 3_choco --dataset_type dataset2 --learning_rate 3e-4 --batch_size 16 --patience 30 --min_delta 0.001


#python main.py --model_type video_tf --experiment 1 --dataset_type dataset2 --learning_rate 3e-4 --batch_size 16 --patience 20 --min_delta 0.001 --test_only --checkpoint_path ./checkpoints/video_tf_hardness1Data_exp1_20250707_224953/best_model.pth
