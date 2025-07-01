import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # <-- 添加这一行
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from timm import create_model

# Import from functions.py
from functions import (
    Dataset_CRNN, EncoderCNN, DecoderRNN, DecoderGRU,
    DecoderTransformer, DecoderTCN, VideoTransformer, Model
)

# --- Configuration ---
# Default values, can be overridden by args or modified here
DEF_DATA_PATH_1 = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness_5_2024_03_03"
DEF_DATA_PATH_2 = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness-1Data"  # Second dataset path

DATASET_OPTIONS = {
    "dataset1": {
        "path": DEF_DATA_PATH_1,
        "name": "hardness5"
    },
    "dataset2": {
        "path": DEF_DATA_PATH_2,
        "name": "hardness1Data"
    }
}

DEF_CHECKPOINT_DIR = "./checkpoints"
DEF_RESULTS_DIR = "./results"

# Architecture Params (from original hardness_CRNN_fixed.py)
CNN_FC_HIDDEN1, CNN_FC_HIDDEN2 = 512, 348
CNN_EMBED_DIM = 256
IMG_X, IMG_Y = 224, 224  # Image dimensions
DROPOUT_P = 0
# RNN/Transformer/TCN Params
SEQ_LEN = 10  # Number of frames per sample, crucial for PositionalEncoding and TCN
RNN_HIDDEN_LAYERS = 1
RNN_HIDDEN_NODES = 256
RNN_FC_DIM = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUMLAYERS = 2
TCN_NUM_LEVELS = 3
TCN_KERNEL_SIZE = 3

# Training Params
EPOCHS = 400
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOG_INTERVAL = 25  # Print training log every N batches
RANDOM_STATE = 42  # For reproducible splits


# --- Helper Functions ---
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"**Display: Using device: {device}**")
    return device


def create_dirs(model_name, dataset_name):
    """Create directories for saving model checkpoints and results, specific to dataset and model"""
    # Combine model_name and dataset_name for unique directory structure
    model_dataset_name = f"{model_name}_{dataset_name}"

    # Create all necessary directories
    checkpoint_dir = os.path.join(DEF_CHECKPOINT_DIR, model_dataset_name)
    logs_dir = os.path.join(DEF_RESULTS_DIR, "logs", model_dataset_name)
    plots_dir = os.path.join(DEF_RESULTS_DIR, "plots", model_dataset_name)
    predictions_dir = os.path.join(DEF_RESULTS_DIR, "predictions", model_dataset_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    print(f"**Display: Created directories for model '{model_name}' with dataset '{dataset_name}'**")

    return {
        "checkpoint_dir": checkpoint_dir,
        "logs_dir": logs_dir,
        "plots_dir": plots_dir,
        "predictions_dir": predictions_dir
    }


# def get_data_loaders(data_path, batch_size, random_state, img_x, img_y):
#     print(f"**Display: Loading and splitting data from: {data_path}**")
#     all_Y_list_raw = []
#     all_X_list_fnames = []
#     # Filter out system files like .DS_Store and ensure correct sorting if needed
#     fnames = sorted(
#         [f for f in os.listdir(data_path) if not f.startswith('.') and os.path.isdir(os.path.join(data_path, f))])
#
#     for f_dir in fnames:
#         try:
#             label = int(f_dir[7:9])  # Assuming folder name format like 'hard_xx_...'
#             all_Y_list_raw.append(label)
#             all_X_list_fnames.append(f_dir)
#         except ValueError:
#             print(f"Warning: Could not parse label from folder name '{f_dir}'. Skipping.")
#             continue
#
#
#     if not all_X_list_fnames:
#         raise ValueError(f"No valid data folders found in {data_path}. Check path and folder naming.")
def get_data_loaders(data_path, batch_size, random_state, img_x, img_y, dataset_type, n_frames=10):
    print(f"**Display: Loading and splitting data from: {data_path}**")
    all_Y_list_raw = []
    all_X_list_fnames = []

        # Filter out system files like .DS_Store and ensure correct sorting
    fnames = sorted(
        [f for f in os.listdir(data_path) if not f.startswith('.') and os.path.isdir(os.path.join(data_path, f))])

    # Set label position based on dataset type
    if dataset_type == 'dataset1':  # hardness5
        label_start_idx = 7  # 'hard_XX_...'
    elif dataset_type == 'dataset2':  # hardness10
        label_start_idx = 8  # different format with label at position 8-9
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    label_end_idx = label_start_idx + 2  # Extract 2 digits

    for f_dir in fnames:
        try:
            # Extract label based on dataset-specific position
            if len(f_dir) > label_end_idx:
                label_str = f_dir[label_start_idx:label_end_idx]
                label = int(label_str)
                all_Y_list_raw.append(label)
                all_X_list_fnames.append(f_dir)
            else:
                print(f"Warning: Folder name '{f_dir}' too short for label extraction. Skipping.")
                continue
        except ValueError:
            print(
                f"Warning: Could not parse label from folder name '{f_dir}' at position {label_start_idx}:{label_end_idx}. Skipping.")
            continue

    if not all_X_list_fnames:
        raise ValueError(f"No valid data folders found in {data_path}. Check path and folder naming.")

    all_Y_list_raw = np.array(all_Y_list_raw, dtype=np.float32).reshape(-1, 1)

    # Split: 60% train, 20% validation, 20% test
    X_train_val, X_test, Y_train_val_raw, Y_test_raw = train_test_split(
        all_X_list_fnames, all_Y_list_raw, test_size=0.20, random_state=random_state
    )
    X_train, X_val, Y_train_raw, Y_val_raw = train_test_split(
        X_train_val, Y_train_val_raw, test_size=0.25, random_state=random_state  # 0.25 * 0.80 = 0.20
    )

    print(f"**Display: Dataset split: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)} samples.**")

    scaler = StandardScaler()
    Y_train_scaled = scaler.fit_transform(Y_train_raw)
    Y_val_scaled = scaler.transform(Y_val_raw)
    Y_test_scaled = scaler.transform(Y_test_raw)

    scale_mean = scaler.mean_.astype(np.float32)
    scale_sigma = scaler.scale_.astype(np.float32)
    print(f"**Display: Labels scaled. Mean: {scale_mean[0]:.2f}, Sigma: {scale_sigma[0]:.2f} (from training data)**")

    transform = transforms.Compose([
        transforms.Resize([img_x, img_y]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Dataset_CRNN(data_path, X_train, Y_train_scaled, Y_train_raw, transform=transform, n_frames=n_frames)
    val_dataset = Dataset_CRNN(data_path, X_val, Y_val_scaled, Y_val_raw, transform=transform, n_frames=n_frames)
    test_dataset = Dataset_CRNN(data_path, X_test, Y_test_scaled, Y_test_raw, transform=transform, n_frames=n_frames)

    # Dataloader params from original code
    loader_params = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': False}  # pin_memory True if using CUDA

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)

    return train_loader, val_loader, test_loader, scaler


def get_model(model_type: str, n_frames, device):
    if model_type == 'video_tf':
        # print(f"**Display: Initializing VideoTransformer model with {n_frames} frames per sample.**")
        model = VideoTransformer(
            img_size=IMG_X,
            patch_size=16,  # Assuming patch size of 16 for ViT
            n_frames=n_frames,
            embed_dim=128,
            depth=6,  # Number of transformer layers
            n_head=4,  # Number of attention heads
            mlp_ratio=4.0,  # MLP ratio for transformer
            drop_p=DROPOUT_P
        ).to(device)
        return model
    # --- Encoder (始终在主设备) ---
    cnn_encoder = EncoderCNN(
        img_x=224, img_y=224,
        fc_hidden1=512,
        fc_hidden2=348,
        drop_p=0.1,
        CNN_embed_dim=256
    ).to(device)

    # --- Decoder (根据模型类型放置设备，但超参保持不变) ---
    if model_type == 'lstm':
        decoder_device = torch.device('cpu')
        decoder = DecoderRNN(
            CNN_embed_dim=256,
            h_RNN_layers=1,
            h_RNN=256,
            h_FC_dim=128,
            drop_p=0.1
        ).to(decoder_device)
    elif model_type == 'gru':
        decoder = DecoderGRU(
            CNN_embed_dim=256,
            h_RNN_layers=1,
            h_RNN=256,
            h_FC_dim=128,
            drop_p=0.1
        ).to(device)
    elif model_type == 'transformer':
        decoder = DecoderTransformer(
            CNN_embed_dim=256,
            nhead=8,
            num_layers=2,
            h_FC_dim=128,
            drop_p=0.2
        ).to(device)
    elif model_type == 'tcn':
        decoder = DecoderTCN(
            CNN_embed_dim=256,
            num_levels=3,
            h_FC_dim=128,
            kernel_size=5,
            drop_p=0.3
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Model(cnn_encoder, decoder, model_type)


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, scaler, log_interval
                # global_step, warmup_steps, base_lr, main_scheduler
                ):
    """
    返 回:
        avg_epoch_loss, avg_epoch_mae_unscaled, global_step
    """
    model.train()

    total_loss, total_abs_err_unscaled, total_samples = 0.0, 0.0, 0
    # 判定 label 所在设备
    target_device = (next(model.Decoder.parameters()).device
                     if hasattr(model, 'Decoder') else device)

    pbar = tqdm(enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for batch_idx, (X, y_scale, y_orig) in pbar:
        # -------------------------------------------------
        # 1)  warm-up OR 调度器: 先算目标 lr，稍后真正写入
        # if global_step < warmup_steps:
        #     # --- Warm-up 阶段 ---
        #     # 线性增加学习率
        #     lr_scale = float(global_step + 1) / float(max(1, warmup_steps))
        #     for pg in optimizer.param_groups:
        #         pg['lr'] = base_lr * lr_scale
        # else:
        #     # --- Cosine Annealing 阶段 ---
        #     # Warm-up结束后，由主调度器接管
        #     # 我们在 optimizer.step() 之后调用 scheduler.step()
        #     pass  # 占位，实际操作在 optimizer.step() 之后
        # -------------------------------------------------
        # 2)  前向 + 反向
        # -------------------------------------------------
        X = X.to(device,  dtype=torch.float32)
        y_scale = y_scale.to(target_device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(X)                    # forward
        if output.device != y_scale.device:  # 防御性保障
            output = output.to(y_scale.device)

        loss = criterion(output, y_scale)    # MSE
        rmse_loss = torch.sqrt(loss)         # RMSE

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                     # <-- 必须先优化器 step
        # 检查梯度范数
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # if grad_norm < 0.01:
        #     print(f"**Display: Warning: Gradient norm too small: {grad_norm:.4f}**")

        # -------------------------------------------------
        # 3)  在优化器更新之后，调用调度器
        # -------------------------------------------------
        # if global_step >= warmup_steps:
        #     main_scheduler.step()

        # -------------------------------------------------
        # 4)  统计指标
        # -------------------------------------------------
        pred_unscaled = scaler.inverse_transform(
            output.detach().cpu().numpy()
        )
        abs_err_unscaled = np.abs(pred_unscaled - y_orig.numpy())

        bs = X.size(0)
        total_loss              += rmse_loss.item() * bs
        total_abs_err_unscaled  += np.sum(abs_err_unscaled)
        total_samples           += bs
        # global_step             += 1         # 关键：累加在循环最后

        # 动态进度条
        if (batch_idx+1) % log_interval == 0 or (batch_idx+1) == len(train_loader):
            pbar.set_postfix_str(
                f"RMSE: {total_loss/total_samples:.4f}, "
                f"MAE(unscaled): {total_abs_err_unscaled/total_samples:.2f}"
            )

    avg_epoch_loss         = total_loss / total_samples
    avg_epoch_mae_unscaled = total_abs_err_unscaled / total_samples
    return avg_epoch_loss, avg_epoch_mae_unscaled



def validate_epoch(model, device, val_loader, criterion, scaler, epoch=None):
    model.eval()
    total_loss = 0.0
    total_abs_err_unscaled = 0.0
    total_samples = 0

    # 设备选择逻辑与训练相同
    if hasattr(model, 'Decoder'):
        target_device = next(model.Decoder.parameters()).device
    else:  # VideoTransformer
        target_device = device

    desc_str = "Validation"
    if epoch is not None: desc_str = f"Epoch {epoch + 1}/{EPOCHS} [Val]"

    pbar = tqdm(val_loader, total=len(val_loader), desc=desc_str)
    with torch.no_grad():
        for X, y_scale, y_orig in pbar:
            X = X.to(device, dtype=torch.float32)
            y_scale_target = y_scale.to(target_device, dtype=torch.float32)

            output = model(X)

            # 确保输出和目标在同一设备上
            if output.device != y_scale_target.device:
                output = output.to(y_scale_target.device)

            loss = criterion(output, y_scale_target)
            rmse_loss = torch.sqrt(loss)

            total_loss += rmse_loss.item() * X.size(0)

            pred_unscaled = scaler.inverse_transform(output.detach().cpu().numpy())
            abs_error_unscaled = np.abs(pred_unscaled - y_orig.numpy())
            total_abs_err_unscaled += np.sum(abs_error_unscaled)
            total_samples += X.size(0)

            avg_loss_so_far = total_loss / total_samples
            avg_mae_unscaled_so_far = total_abs_err_unscaled / total_samples
            pbar.set_postfix_str(f"RMSE Loss: {avg_loss_so_far:.4f}, MAE (unscaled): {avg_mae_unscaled_so_far:.2f}")

    avg_epoch_loss = total_loss / total_samples
    avg_epoch_mae_unscaled = total_abs_err_unscaled / total_samples
    print(
        f"**Display: {desc_str} Results: Avg RMSE Loss: {avg_epoch_loss:.4f}, Avg MAE (unscaled): {avg_epoch_mae_unscaled:.2f}**")
    return avg_epoch_loss, avg_epoch_mae_unscaled


def test_model_and_save_results(model, device, test_loader, scaler, model_name, dataset_name, dirs,
                                checkpoint_path=None):
    if checkpoint_path:
        print(f"**Display: Loading model from {checkpoint_path} for testing.**")

        # 检查模型类型以使用正确的加载逻辑
        if hasattr(model, 'decoder'):  # Case 1: Old model structure (Encoder+Decoder)
            print("**Display: Detected legacy model structure (Encoder+Decoder). Using careful loading.**")
            try:
                # 尝试标准加载
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            except RuntimeError as e:
                # 如果失败（例如LSTM在CPU上），则使用特殊加载
                print(f"Standard load failed ({e}), attempting careful load for mixed devices (e.g. LSTM)...")
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
                # 确保模块在加载后位于其指定设备上
                model.encoder.to(device)
                if model.model_type == 'lstm':
                    model.decoder.to(torch.device('cpu'))
                else:
                    model.decoder.to(device)
        else:  # Case 2: New standalone model (like VideoTransformer)
            print("**Display: Detected standalone model structure (e.g., VideoTransformer). Using standard loading.**")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        print(f"**Display: Model loaded from {checkpoint_path} successfully.**")

    model.eval()
    all_predictions_unscaled = []
    all_true_values_unscaled = []
    all_scaled_predictions = []
    all_scaled_true = []

    # decoder_param_device = ... # <-- 已删除：此变量不再需要且会导致错误

    print("**Display: Starting testing phase...**")
    pbar = tqdm(test_loader, total=len(test_loader), desc="Testing")
    with torch.no_grad():
        for X, y_scale, y_orig in pbar:
            X = X.to(device, dtype=torch.float32)
            output_scaled = model(X)

            pred_unscaled = scaler.inverse_transform(output_scaled.detach().cpu().numpy())

            all_predictions_unscaled.extend(pred_unscaled.flatten().tolist())
            all_true_values_unscaled.extend(y_orig.numpy().flatten().tolist())
            all_scaled_predictions.extend(output_scaled.detach().cpu().numpy().flatten().tolist())
            all_scaled_true.extend(y_scale.numpy().flatten().tolist())

    # --- 后续的指标计算和保存部分无需修改，保持原样即可 ---
    # Calculate metrics
    true_np = np.array(all_true_values_unscaled)
    pred_np = np.array(all_predictions_unscaled)

    mse = mean_squared_error(true_np, pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_np, pred_np)
    r2 = r2_score(true_np, pred_np)

    non_zero_mask = true_np != 0
    if np.sum(non_zero_mask) == 0:
        mre = float('inf')
    else:
        mre = np.mean(np.abs((pred_np[non_zero_mask] - true_np[non_zero_mask]) / true_np[non_zero_mask]))

    print("\n**Display: Test Results:**")
    print(f"  **RMSE (unscaled): {rmse:.4f}**")
    print(f"  **MAE (unscaled): {mae:.4f}**")
    print(f"  **R2 Score: {r2:.4f}**")
    print(f"  **Mean Relative Error (MRE): {mre * 100:.2f}%**")

    # Save results
    metrics_summary = (
        f"Test Results for model: {model_name}, dataset: {dataset_name}\n"
        f"Checkpoint: {checkpoint_path or 'N/A (trained in same run)'}\n"
        f"RMSE (unscaled): {rmse:.4f}\n"
        f"MAE (unscaled): {mae:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Mean Relative Error (MRE): {mre * 100:.2f}%\n"
        f"Number of test samples: {len(true_np)}\n"
    )
    metrics_path = os.path.join(dirs["predictions_dir"], f"{model_name}_{dataset_name}_test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_summary)
    print(f"**Display: Test metrics saved to {metrics_path}**")

    df_predictions = pd.DataFrame({
        'true_value_unscaled': all_true_values_unscaled,
        'predicted_value_unscaled': all_predictions_unscaled,
        'true_value_scaled': all_scaled_true,
        'predicted_value_scaled': all_scaled_predictions
    })
    predictions_path = os.path.join(dirs["predictions_dir"], f"{model_name}_{dataset_name}_test_predictions.csv")
    df_predictions.to_csv(predictions_path, index=False)
    print(f"**Display: Test predictions saved to {predictions_path}**")

    return rmse, mae, r2, mre


def plot_metrics(epochs_ran, train_losses, val_losses, train_errors, val_errors, model_name, dataset_name, dirs):
    epochs_axis = np.arange(1, epochs_ran + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis, train_losses, label='Train RMSE Loss')
    plt.plot(epochs_axis, val_losses, label='Validation RMSE Loss')
    plt.title(f'{model_name.upper()} - {dataset_name} - Model RMSE Loss')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_axis, train_errors, label='Train MAE (unscaled)')
    plt.plot(epochs_axis, val_errors, label='Validation MAE (unscaled)')
    plt.title(f'{model_name.upper()} - {dataset_name} - Model MAE (unscaled)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(dirs["plots_dir"], f"{model_name}_{dataset_name}_training_curves.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"**Display: Training curves plot saved to {plot_filename}**")



def main(args):
    # Get dataset information based on selection
    dataset_info = DATASET_OPTIONS.get(args.dataset_type)
    if not dataset_info:
        raise ValueError(
            f"Invalid dataset type: {args.dataset_type}. Valid options are: {', '.join(DATASET_OPTIONS.keys())}")

    data_path = dataset_info["path"]
    dataset_name = dataset_info["name"]

    print(f"**Display: Using dataset: {dataset_name} from {data_path}**")

    # Setup
    device = setup_device()

    # Create dataset and model specific directories
    dirs = create_dirs(args.model_type, dataset_name)

    # Data
    train_loader, val_loader, test_loader, scaler = get_data_loaders(
        data_path, args.batch_size, RANDOM_STATE, IMG_X, IMG_Y, args.dataset_type, args.n_frames
    )
    # ========================================================================
    # ### 1. 初始化 TensorBoard Writer (在创建目录后) ###
    # 它会在你的 logs_dir 中创建一个新的 'runs' 子目录来存放日志
    writer = SummaryWriter(log_dir=dirs["logs_dir"])
    # ========================================================================

    # Model
    model = get_model(args.model_type, args.n_frames, device)
    # 冻结早期层
    # for name, p in model.named_parameters():
    #     if not name.startswith(("temporal_transformer.layers.1", "head")):
    #         p.requires_grad = False

    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    # optimizer = optim.AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=3e-4,  # 调低学习率以保留预训练特征
    #     weight_decay=0
    # )
    criterion = nn.MSELoss()  # We will take sqrt for RMSE as in original code

    # 5% warm-up + cosine
    # 2. 设置带Warm-up的余弦退火调度器
    # total_steps = len(train_loader) * args.epochs
    # warmup_percentage = 0.1  # 使用10%的步数进行warm-up
    # warmup_steps = int(total_steps * warmup_percentage)
    #
    # print(f"**Display: LR Scheduler enabled. Total steps: {total_steps}, Warm-up steps: {warmup_steps}.**")
    # main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_steps - warmup_steps  # T_max是余弦退火的总步数
    # )

    # Training Loop
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_errors, epoch_val_errors = [], []  # Using MAE (unscaled) for error tracking
    # global_step = 0  # Initialize global step for scheduler

    print(
        f"\n**Display: Starting training for {args.model_type.upper()} model on {dataset_name} for {args.epochs} epochs...**")
    for epoch in range(args.epochs):
        # train_loss, train_error, global_step = train_epoch(
        #     model, device, train_loader, optimizer, criterion, epoch,
        #     global_step=global_step, main_scheduler=main_scheduler,
        #     warmup_steps=warmup_steps, base_lr=args.learning_rate,
        #     scaler=scaler, log_interval=args.log_interval
        # )
        train_loss, train_error = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch,
            scaler=scaler, log_interval=args.log_interval
            # global_step=global_step, warmup_steps=warmup_steps, base_lr=args.learning_rate, main_scheduler=main_scheduler
        )
        val_loss, val_error = validate_epoch(
            model, device, val_loader, criterion, scaler, epoch
        )

        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_train_errors.append(train_error)
        epoch_val_errors.append(val_error)

        # ========================================================================
        # ### 2. 在每个 epoch 后将数据写入 TensorBoard ###
        # 记录损失 (将训练和验证损失画在同一张图上)
        writer.add_scalar('Loss/Train_RMSE', train_loss, epoch)
        writer.add_scalar('Loss/Validation_RMSE', val_loss, epoch)

        # 记录误差 (将训练和验证误差画在另一张图上)
        writer.add_scalar('Error/Train_MAE_Unscaled', train_error, epoch)
        writer.add_scalar('Error/Validation_MAE_Unscaled', val_error, epoch)

        # 记录学习率 (如果未来加回调度器，这会很有用)
        # writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        # ========================================================================

        # Save model checkpoint every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            print(f"**Display: Saving checkpoint for epoch {epoch + 1}...**")
            checkpoint_name = f"{args.model_type}_{dataset_name}_epoch_{epoch + 1}.pth"
            checkpoint_save_path = os.path.join(dirs["checkpoint_dir"], checkpoint_name)
            torch.save(model.state_dict(), checkpoint_save_path)

        # Save logs after each epoch
        log_data = {
            'train_rmse_loss': epoch_train_losses,
            'val_rmse_loss': epoch_val_losses,
            'train_mae_unscaled': epoch_train_errors,
            'val_mae_unscaled': epoch_val_errors
        }
        df_logs = pd.DataFrame(log_data)
        log_path = os.path.join(dirs["logs_dir"], f"{args.model_type}_{dataset_name}_training_log.csv")
        df_logs.to_csv(log_path, index_label="epoch")

    print("**Display: Training finished.**")
    # ========================================================================
    # ### 3. 训练结束后关闭 writer ###
    writer.close()
    print("**Display: TensorBoard log saved.**")
    # ========================================================================

    # Plot training curves
    plot_metrics(args.epochs, epoch_train_losses, epoch_val_losses,
                 epoch_train_errors, epoch_val_errors, args.model_type, dataset_name, dirs)

    # Test the model using the final checkpoint
    final_model_path = os.path.join(dirs["checkpoint_dir"], f"{args.model_type}_{dataset_name}_epoch_{args.epochs}.pth")
    print(f"\n**Display: Proceeding to test the final model from epoch {args.epochs} on {dataset_name}...**")
    test_model_and_save_results(model, device, test_loader, scaler, args.model_type, dataset_name, dirs,
                                checkpoint_path=final_model_path)

    print(f"**Display: Process for model {args.model_type.upper()} on dataset {dataset_name} completed.**")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Evaluate CRNN-variant models for hardness estimation.")
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'gru', 'transformer', 'tcn', 'video_tf'],
                        help="Type of decoder model to use (lstm, gru, transformer, tcn).")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['dataset1', 'dataset2'],
                        help="Which dataset to use (dataset1: hardness5, dataset2: hardness10)")
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS}).")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f"Batch size for training and evaluation (default: {BATCH_SIZE}).")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f"Learning rate for the optimizer (default: {LEARNING_RATE}).")
    parser.add_argument('--log_interval', type=int, default=LOG_INTERVAL,
                        help=f"Log training progress every N batches (default: {LOG_INTERVAL}).")
    parser.add_argument('--n_frames', type=int, default=10, help="Number of frames per sample (default: 10).")

    parsed_args = parser.parse_args()

    # Ensure globals are updated
    EPOCHS = parsed_args.epochs
    BATCH_SIZE = parsed_args.batch_size
    LEARNING_RATE = parsed_args.learning_rate
    LOG_INTERVAL = parsed_args.log_interval
    N_FRAMES = parsed_args.n_frames

    main(parsed_args)
