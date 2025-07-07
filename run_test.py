import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# 确保能从 functions.py 导入所需模块
try:
    from functions import (
        Dataset_CRNN, EncoderCNN, DecoderRNN, DecoderGRU,
        DecoderTransformer, DecoderTCN, VideoTransformer, Model
    )
except ImportError:
    print("错误：请确保 `functions.py` 文件与此脚本在同一目录下。")
    exit()

# ===================================================================
# ---      配置参数 (必须与 train_eval.py 保持完全一致)         ---
# ===================================================================
DEF_DATA_PATH_1 = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness_5_2024_03_03"
DEF_DATA_PATH_2 = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness-1Data"

DATASET_OPTIONS = {
    "dataset1": {"path": DEF_DATA_PATH_1, "name": "hardness5"},
    "dataset2": {"path": DEF_DATA_PATH_2, "name": "hardness1Data"}
}

DEF_CHECKPOINT_DIR = "./checkpoints"
DEF_RESULTS_DIR = "./results"
IMG_X, IMG_Y = 224, 224
RANDOM_STATE = 42  # **关键：固定的随机种子确保测试集始终相同**
GROUPS_TO_FIND = ['FLAT', 'EDGE', 'CORNER', 'SPHERE', 'CYLIND', 'BASIC', 'SHELL', 'CHOCO']


# ===================================================================
# ---      辅助函数 (从 train_eval.py 精确复制而来)             ---
# ===================================================================

def setup_device():
    """选择可用的最佳设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"**Display: 使用设备: {device}**")
    return device


def create_dirs(model_name, dataset_name):
    """创建用于保存结果的目录，逻辑与训练脚本一致"""
    model_dataset_name = f"{model_name}_{dataset_name}"
    # 即使只测试，也创建所有目录以保持路径结构统一
    checkpoint_dir = os.path.join(DEF_CHECKPOINT_DIR, model_dataset_name)
    logs_dir = os.path.join(DEF_RESULTS_DIR, "logs", model_dataset_name)
    plots_dir = os.path.join(DEF_RESULTS_DIR, "plots", model_dataset_name)
    predictions_dir = os.path.join(DEF_RESULTS_DIR, "predictions", model_dataset_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    print(f"**Display: 已确保 '{model_name}' 在 '{dataset_name}' 上的结果目录存在。**")
    return {
        "checkpoint_dir": checkpoint_dir, "logs_dir": logs_dir,
        "plots_dir": plots_dir, "predictions_dir": predictions_dir
    }


def get_data_loaders(data_path, batch_size, random_state, img_x, img_y, dataset_type, n_frames):
    """
    精确复制的数据加载和分割函数。
    它会重新进行数据分割，但由于 random_state 固定，test set 是相同的。
    它会返回测试集加载器 (test_loader) 和在训练集上拟合好的标准化器 (scaler)。
    """
    print(f"**Display: 加载并分割数据来源: {data_path}**")
    all_Y_list_raw, all_X_list_fnames, all_X_groups = [], [], []
    fnames = sorted(
        [f for f in os.listdir(data_path) if not f.startswith('.') and os.path.isdir(os.path.join(data_path, f))])

    label_start_idx = 7 if dataset_type == 'dataset1' else 8
    label_end_idx = label_start_idx + 2
    for f_dir in fnames:
        try:
            label = int(f_dir[label_start_idx:label_end_idx])
            all_Y_list_raw.append(label)
            all_X_list_fnames.append(f_dir)

            # 从文件夹名称中解析组别
            group_found = "UNKNOWN"
            for group_name in GROUPS_TO_FIND:
                if group_name in f_dir.upper():
                    group_found = group_name
                    break
            all_X_groups.append(group_found)
        except (ValueError, IndexError):
            continue

    if not all_X_list_fnames: raise ValueError(f"在 {data_path} 中未找到有效数据文件夹。")

    all_Y_list_raw = np.array(all_Y_list_raw, dtype=np.float32).reshape(-1, 1)

    # 关键分割步骤：与训练脚本完全一致
    X_train_val, X_test, Y_train_val_raw, Y_test_raw, G_train_val, G_test = train_test_split(
        all_X_list_fnames, all_Y_list_raw, all_X_groups, test_size=0.20, random_state=random_state)
    X_train, _, Y_train_raw, _ = train_test_split(X_train_val, Y_train_val_raw, test_size=0.25,
                                                  random_state=random_state)

    # 关键标准化步骤：Scaler只在训练数据上拟合
    scaler = StandardScaler()
    scaler.fit(Y_train_raw)
    Y_test_scaled = scaler.transform(Y_test_raw)

    transform = transforms.Compose([
        transforms.Resize([img_x, img_y]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = Dataset_CRNN(data_path, X_test, Y_test_scaled, Y_test_raw, transform=transform, n_frames=n_frames, groups=G_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(f"**Display: 测试集加载器和标准化器准备就绪。测试样本数: {len(test_dataset)}**")
    return test_loader, scaler


def get_model(model_type, n_frames, device, args):
    """
    模型重建函数。必须使用与训练时完全相同的参数来实例化模型。
    """
    if model_type == 'video_tf':
        model = VideoTransformer(
            img_size=IMG_X,
            patch_size=args.vit_patch_size,
            n_frames=n_frames,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            n_head=args.vit_n_head,
            mlp_ratio=4.0,  # 保持一致
            drop_p=0.1  # 保持一致
        ).to(device)
        return model
    else:
        # 为其他旧模型保留的逻辑
        cnn_encoder = EncoderCNN(img_x=IMG_X, img_y=IMG_Y, CNN_embed_dim=256).to(device)
        if model_type == 'gru':
            decoder = DecoderGRU(CNN_embed_dim=256).to(device)
        elif model_type == 'lstm':
            decoder = DecoderRNN(CNN_embed_dim=256).to(torch.device('cpu'))
        else:
            raise ValueError(f"此测试脚本不支持模型类型: {model_type}")
        return Model(cnn_encoder, decoder, model_type)


def test_model_and_save_results(model, device, test_loader, scaler, model_name, dataset_name, dirs, checkpoint_path):
    print(f"**Display: 从 {checkpoint_path} 加载模型权重...**")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"**Display: 模型权重加载成功。**")
    model.eval()

    # 使用 defaultdict 来自动处理新遇到的组别
    results_by_group = defaultdict(lambda: {'preds': [], 'trues': []})

    # 用于最终CSV文件的列表
    all_preds_unscaled, all_trues_unscaled, all_preds_scaled, all_trues_scaled, all_groups_for_csv = [], [], [], [], []

    print("**Display: 开始在测试集上进行评估...**")
    pbar = tqdm(test_loader, total=len(test_loader), desc="Testing")
    with torch.no_grad():
        # Dataloader 现在返回4个值
        for X, y_scale, y_orig, groups_batch in pbar:
            X = X.to(device, dtype=torch.float32)
            output_scaled = model(X)
            pred_unscaled = scaler.inverse_transform(output_scaled.detach().cpu().numpy())

            # 按样本处理，以便匹配组别
            for i in range(len(X)):
                group_name = groups_batch[i]
                pred_val = pred_unscaled[i, 0]
                true_val = y_orig[i, 0]

                # 收集每个组别的数据
                results_by_group[group_name]['preds'].append(pred_val)
                results_by_group[group_name]['trues'].append(true_val)

                # 为CSV文件收集所有数据
                all_preds_unscaled.append(pred_val)
                all_trues_unscaled.append(true_val)
                all_preds_scaled.append(output_scaled.detach().cpu().numpy()[i, 0])
                all_trues_scaled.append(y_scale.numpy()[i, 0])
                all_groups_for_csv.append(group_name)

    print("\n" + "=" * 50)
    print("           按组别划分的测试结果")
    print("=" * 50)

    # 定义一个辅助函数来计算和打印指标，避免代码重复
    def calculate_and_print_metrics(group_name, trues, preds):
        if not trues:
            print(f"\n--- 组别: {group_name} (0 个样本) ---")
            return

        true_np, pred_np = np.array(trues), np.array(preds)
        rmse = np.sqrt(mean_squared_error(true_np, pred_np))
        mae = mean_absolute_error(true_np, pred_np)
        r2 = r2_score(true_np, pred_np)

        print(f"\n--- 组别: {group_name} ({len(trues)} 个样本) ---")
        print(f"  **RMSE (unscaled): {rmse:.4f}**")
        print(f"  **MAE (unscaled): {mae:.4f}**")
        print(f"  **R² Score: {r2:.4f}**")

    # 为每个组别计算并打印结果
    for group in sorted(results_by_group.keys()):
        data = results_by_group[group]
        calculate_and_print_metrics(group, data['trues'], data['preds'])

    # 打印总结果
    print("\n" + "=" * 50)
    print("               总体测试结果")
    print("=" * 50)
    calculate_and_print_metrics("OVERALL", all_trues_unscaled, all_preds_unscaled)
    print("=" * 50 + "\n")

    # 保存包含组别信息的详细预测结果
    df_predictions = pd.DataFrame({
        'group': all_groups_for_csv,
        'true_value_unscaled': all_trues_unscaled,
        'predicted_value_unscaled': all_preds_unscaled,
        'true_value_scaled': all_trues_scaled,
        'predicted_value_scaled': all_preds_scaled,
    })
    predictions_path = os.path.join(dirs["predictions_dir"],
                                    f"{model_name}_{dataset_name}_test_predictions_by_group.csv")
    df_predictions.to_csv(predictions_path, index=False)
    print(f"**Display: 按组别划分的详细预测结果已保存至: {predictions_path}**")
# ===================================================================
# ---                         主程序入口                          ---
# ===================================================================

def main(args):
    """主函数，编排测试流程"""
    print("--- 独立模型测试脚本 ---")

    device = setup_device()
    dataset_info = DATASET_OPTIONS[args.dataset_type]
    data_path, dataset_name = dataset_info["path"], dataset_info["name"]

    # 1. 准备目录、数据加载器和标准化器
    dirs = create_dirs(args.model_type, dataset_name)
    test_loader, scaler = get_data_loaders(data_path, args.batch_size, RANDOM_STATE, IMG_X, IMG_Y, args.dataset_type,
                                           args.n_frames)

    # 2. 根据命令行参数重建模型架构
    print("\n**Display: 正在根据提供的参数重建模型架构...**")
    model = get_model(args.model_type, args.n_frames, device, args)

    # 3. 运行测试并保存结果
    print("\n**Display: 开始执行评估...**")
    test_model_and_save_results(model, device, test_loader, scaler, args.model_type, dataset_name, dirs,
                                args.checkpoint_path)

    print("\n--- 测试完成 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="用于测试已保存模型检查点的独立脚本")

    # --- 关键输入 ---
    parser.add_argument('--checkpoint_path', type=str, required=True, help="要测试的已保存模型检查点(.pth)的路径。")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['lstm', 'gru', 'transformer', 'tcn', 'video_tf'], help="模型的架构类型。")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['dataset1', 'dataset2'],
                        help="使用的数据集类型。")

    # --- 数据和模型架构参数 (必须与训练时完全一致) ---
    parser.add_argument('--n_frames', type=int, default=10, help="每个样本的帧数。")
    parser.add_argument('--batch_size', type=int, default=16, help="评估时使用的批大小。")

    # VideoTransformer 专用参数
    parser.add_argument('--vit_embed_dim', type=int, default=128, help="VideoTransformer的嵌入维度。")
    parser.add_argument('--vit_depth', type=int, default=2, help="VideoTransformer的深度（层数）。")
    parser.add_argument('--vit_n_head', type=int, default=2, help="VideoTransformer的注意力头数。")
    parser.add_argument('--vit_patch_size', type=int, default=16, help="VideoTransformer的Patch大小。")

    parsed_args = parser.parse_args()
    main(parsed_args)
