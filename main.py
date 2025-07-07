import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from timm import create_model
import re
import time
from datetime import datetime

# Import from functions.py (假设该文件存在且包含必要的类定义)
# 如果 functions.py 不存在，请确保下面的类定义在某个地方
try:
    from functions import (
        Dataset_CRNN, EncoderCNN, DecoderRNN, DecoderGRU,
        DecoderTransformer, DecoderTCN, VideoTransformer, Model
    )
except ImportError:
    print("Warning: 'functions.py' not found. Ensure required class definitions are available.")


    # 如果没有 functions.py，你需要在这里或另一个文件中定义这些类
    # 例如: class Dataset_CRNN(torch.utils.data.Dataset): ...
    # 为了代码能跑通，这里先放一个占位符
    class Placeholder:
        pass


    Dataset_CRNN, EncoderCNN, DecoderRNN, DecoderGRU, DecoderTransformer, DecoderTCN, VideoTransformer, Model = [
                                                                                                                    Placeholder] * 8

# --- Configuration ---
# 数据集路径 (请根据您的实际路径修改)
DEF_DATA_PATH_1 = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness_5_2024_03_03"
DEF_DATA_PATH_2 = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness-1Data"

DATASET_OPTIONS = {
    "dataset1": {"path": DEF_DATA_PATH_1, "name": "hardness5"},
    "dataset2": {"path": DEF_DATA_PATH_2, "name": "hardness1Data"}
}

# 结果和模型保存的根目录
DEF_CHECKPOINT_DIR = "./checkpoints"
DEF_RESULTS_DIR = "./results"
SHAPES_ALL = ['FLAT', 'EDGE', 'CORNER', 'SPHERE', 'CYLIND', 'BASIC', 'CHOCO']

# 模型和训练超参数
CNN_FC_HIDDEN1, CNN_FC_HIDDEN2 = 512, 348
CNN_EMBED_DIM = 256
IMG_X, IMG_Y = 224, 224
DROPOUT_P = 0
SEQ_LEN = 10
RNN_HIDDEN_LAYERS = 1
RNN_HIDDEN_NODES = 256
RNN_FC_DIM = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUMLAYERS = 2
TCN_NUM_LEVELS = 3
TCN_KERNEL_SIZE = 3
EPOCHS = 400
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOG_INTERVAL = 25
RANDOM_STATE = 42


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


def create_dirs(model_name, dataset_name, experiment):
    """
    根据模型、数据集、实验号和时间戳创建唯一的目录
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = f"{model_name}_{dataset_name}_exp{experiment}_{timestamp}"

    checkpoint_dir = os.path.join(DEF_CHECKPOINT_DIR, run_tag)
    logs_dir = os.path.join(DEF_RESULTS_DIR, "logs", run_tag)
    plots_dir = os.path.join(DEF_RESULTS_DIR, "plots", run_tag)
    predictions_dir = os.path.join(DEF_RESULTS_DIR, "predictions", run_tag)

    # 递归创建所有目录
    for d in [checkpoint_dir, logs_dir, plots_dir, predictions_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"**Display: Created unique directories for this run under tag: {run_tag}**")

    return {
        "checkpoint_dir": checkpoint_dir,
        "logs_dir": logs_dir,
        "plots_dir": plots_dir,
        "predictions_dir": predictions_dir,
        "run_tag": run_tag
    }


def parse_dir_info(dirname:str, dataset_type:str):
    """
    返回 (shape:str, hardness:int)
    dirname 例子:
        FLAT_hard_17_xxx
        CYLIND5_hard_42_xxx
        hard_35_FLAT_xxx  (后两种写法也兼容)
    """
    # 正则表达式匹配形状和硬度
    if dataset_type == 'dataset1':  # hardness5
        pattern = r'(\w+)_hard_(\d{2})'
    elif dataset_type == 'dataset2':  # hardness10
        pattern = r'shore00_(\d{2})-(\w+)_\d+'
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    match = re.search(pattern, dirname)
    if match:
        shape = match.group(2)
        hardness = int(match.group(1))
        return shape, hardness
    else:
        raise ValueError(f"Could not parse shape and hardness from directory name: {dirname}")



def build_exp_dataloaders(data_path, dataset_type, experiment, batch_size, random_state, img_x, img_y, n_frames):
    """
    根据指定的实验号来构建训练、验证和测试Dataloader
    """
    print(f"**Display: Building datasets for Experiment '{experiment}'...**")

    X_train_fnames, y_train_labels = [], []
    X_test_fnames, y_test_labels = [], []

    # 1. 扫描所有数据文件夹并根据实验规则进行分配
    all_dirs = sorted(
        [d for d in os.listdir(data_path) if not d.startswith('.') and os.path.isdir(os.path.join(data_path, d))])

    for dirname in all_dirs:
        shape, hardness = parse_dir_info(dirname, dataset_type)
        if hardness == -1 or shape == 'UNKNOWN':
            print(f"Warning: Could not parse info from '{dirname}'. Skipping.")
            continue

        is_basic_shape = shape in ['FLAT', 'EDGE', 'CORNER', 'SPHERE', 'CYLIND']

        # 根据实验号分配数据
        if experiment == '1':
            if is_basic_shape:
                if hardness == 36:
                    X_test_fnames.append(dirname)
                    y_test_labels.append(hardness)
                else:
                    X_train_fnames.append(dirname)
                    y_train_labels.append(hardness)
        elif experiment == '2':
            if shape == 'CYLIND':
                X_test_fnames.append(dirname)
                y_test_labels.append(hardness)
            elif shape in ['FLAT', 'EDGE', 'CORNER', 'SPHERE']:
                X_train_fnames.append(dirname)
                y_train_labels.append(hardness)
        elif experiment == '3_basic':
            if shape == 'BASIC':
                X_test_fnames.append(dirname)
                y_test_labels.append(hardness)
            elif is_basic_shape:
                X_train_fnames.append(dirname)
                y_train_labels.append(hardness)
        elif experiment == '3_choco':
            if shape == 'CHOCO':
                X_test_fnames.append(dirname)
                y_test_labels.append(hardness)
            elif is_basic_shape:
                X_train_fnames.append(dirname)
                y_train_labels.append(hardness)

    if not X_train_fnames:
        raise ValueError("Training data is empty. Check experiment setup and data paths.")
    if not X_test_fnames:
        print("Warning: Test data is empty for this experiment setup.")

    # 2. 从收集到的训练集中划分出验证集 (80% train, 20% validation)
    X_train_fnames, X_val_fnames, y_train_labels, y_val_labels = train_test_split(
        X_train_fnames, y_train_labels, test_size=0.20, random_state=random_state
    )
    print(
        f"**Display: Dataset split: Train: {len(X_train_fnames)}, Validation: {len(X_val_fnames)}, Test: {len(X_test_fnames)} samples.**")

    # 3. 标准化标签 (Scaler只在训练集上fit)
    y_train_raw = np.array(y_train_labels, dtype=np.float32).reshape(-1, 1)
    y_val_raw = np.array(y_val_labels, dtype=np.float32).reshape(-1, 1)
    y_test_raw = np.array(y_test_labels, dtype=np.float32).reshape(-1, 1) if y_test_labels else np.array([])

    scaler = StandardScaler()
    Y_train_scaled = scaler.fit_transform(y_train_raw)
    Y_val_scaled = scaler.transform(y_val_raw)
    # 只有在测试集不为空时才进行transform
    Y_test_scaled = scaler.transform(y_test_raw) if y_test_raw.size > 0 else np.array([])

    scale_mean = scaler.mean_.astype(np.float32)
    scale_sigma = scaler.scale_.astype(np.float32)
    print(f"**Display: Labels scaled. Mean: {scale_mean[0]:.2f}, Sigma: {scale_sigma[0]:.2f} (from training data)**")

    # 4. 创建 PyTorch Datasets 和 DataLoaders
    transform = transforms.Compose([
        transforms.Resize([img_x, img_y]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Dataset_CRNN(data_path, X_train_fnames, Y_train_scaled, y_train_raw, transform=transform,
                                 n_frames=n_frames)
    val_dataset = Dataset_CRNN(data_path, X_val_fnames, Y_val_scaled, y_val_raw, transform=transform, n_frames=n_frames)
    # 同样，只有在测试集不为空时才创建
    test_dataset = Dataset_CRNN(data_path, X_test_fnames, Y_test_scaled, y_test_raw, transform=transform,
                                n_frames=n_frames) if y_test_raw.size > 0 else None

    loader_params = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': False}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params) if test_dataset else None

    return train_loader, val_loader, test_loader, scaler


# ... (get_model, train_epoch, validate_epoch, test_model_and_save_results, plot_metrics 函数保持不变) ...
# 注意: 为了简洁，这里省略了这些函数的代码，它们与您提供的源代码完全相同。
# 在您的实际文件中，请保留这些函数的完整代码。

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


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, scaler, log_interval):
    model.train()
    total_loss, total_abs_err_unscaled, total_samples = 0.0, 0.0, 0
    target_device = (next(model.Decoder.parameters()).device if hasattr(model, 'Decoder') else device)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

    for batch_idx, batch in pbar:
        X, y_scale, y_orig = batch[:3]
        X = X.to(device, dtype=torch.float32)
        y_scale = y_scale.to(target_device, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(X)
        if output.device != y_scale.device:
            output = output.to(y_scale.device)
        loss = criterion(output, y_scale)
        rmse_loss = torch.sqrt(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pred_unscaled = scaler.inverse_transform(output.detach().cpu().numpy())
        abs_err_unscaled = np.abs(pred_unscaled - y_orig.numpy())
        bs = X.size(0)
        total_loss += rmse_loss.item() * bs
        total_abs_err_unscaled += np.sum(abs_err_unscaled)
        total_samples += bs

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
            pbar.set_postfix_str(
                f"RMSE: {total_loss / total_samples:.4f}, MAE(unscaled): {total_abs_err_unscaled / total_samples:.2f}")

    avg_epoch_loss = total_loss / total_samples
    avg_epoch_mae_unscaled = total_abs_err_unscaled / total_samples
    return avg_epoch_loss, avg_epoch_mae_unscaled


def validate_epoch(model, device, val_loader, criterion, scaler, epoch=None):
    model.eval()
    total_loss, total_abs_err_unscaled, total_samples = 0.0, 0.0, 0
    target_device = next(model.Decoder.parameters()).device if hasattr(model, 'Decoder') else device
    desc_str = "Validation" if epoch is None else f"Epoch {epoch + 1}/{EPOCHS} [Val]"
    pbar = tqdm(val_loader, total=len(val_loader), desc=desc_str)
    with torch.no_grad():
        for batch in pbar:
            X, y_scale, y_orig = batch[:3]
            X = X.to(device, dtype=torch.float32)
            y_scale_target = y_scale.to(target_device, dtype=torch.float32)
            output = model(X)
            if output.device != y_scale_target.device: output = output.to(y_scale_target.device)
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


def test_model_and_save_results(model, device, test_loader, scaler, run_tag, dirs, checkpoint_path=None):
    if checkpoint_path:
        print(f"**Display: Loading model from {checkpoint_path} for testing.**")
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        if hasattr(model, 'encoder'): model.encoder.to(device)
        if hasattr(model, 'decoder'):
            if model.model_type == 'lstm':
                model.decoder.to(torch.device('cpu'))
            else:
                model.decoder.to(device)
        else:  # Standalone model
            model.to(device)
        print(f"**Display: Model loaded successfully.**")

    model.eval()
    all_predictions_unscaled, all_true_values_unscaled = [], []
    all_scaled_predictions, all_scaled_true = [], []

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

    true_np, pred_np = np.array(all_true_values_unscaled), np.array(all_predictions_unscaled)
    mse = mean_squared_error(true_np, pred_np)
    rmse, mae, r2 = np.sqrt(mse), mean_absolute_error(true_np, pred_np), r2_score(true_np, pred_np)
    non_zero_mask = true_np != 0
    mre = np.mean(np.abs((pred_np[non_zero_mask] - true_np[non_zero_mask]) / true_np[non_zero_mask])) if np.sum(
        non_zero_mask) > 0 else float('inf')

    print("\n**Display: Test Results:**")
    print(f"  **RMSE (unscaled): {rmse:.4f}**")
    print(f"  **MAE (unscaled): {mae:.4f}**")
    print(f"  **R2 Score: {r2:.4f}**")
    print(f"  **Mean Relative Error (MRE): {mre * 100:.2f}%**")

    metrics_summary = (
        f"Test Results for run: {run_tag}\n"
        f"Checkpoint: {checkpoint_path or 'N/A (trained in same run)'}\n"
        f"RMSE (unscaled): {rmse:.4f}\nMAE (unscaled): {mae:.4f}\n"
        f"R2 Score: {r2:.4f}\nMean Relative Error (MRE): {mre * 100:.2f}%\n"
        f"Number of test samples: {len(true_np)}\n"
    )
    metrics_path = os.path.join(dirs["predictions_dir"], "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_summary)
    print(f"**Display: Test metrics saved to {metrics_path}**")

    df_predictions = pd.DataFrame({
        'true_value_unscaled': all_true_values_unscaled, 'predicted_value_unscaled': all_predictions_unscaled,
        'true_value_scaled': all_scaled_true, 'predicted_value_scaled': all_scaled_predictions
    })
    predictions_path = os.path.join(dirs["predictions_dir"], "test_predictions.csv")
    df_predictions.to_csv(predictions_path, index=False)
    print(f"**Display: Test predictions saved to {predictions_path}**")


def plot_metrics(epochs_ran, train_losses, val_losses, train_errors, val_errors, run_tag, dirs):
    epochs_axis = np.arange(1, epochs_ran + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis, train_losses, label='Train RMSE Loss')
    plt.plot(epochs_axis, val_losses, label='Validation RMSE Loss')
    plt.title(f'{run_tag} - Model RMSE Loss')
    plt.xlabel('Epochs');
    plt.ylabel('RMSE Loss');
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_axis, train_errors, label='Train MAE (unscaled)')
    plt.plot(epochs_axis, val_errors, label='Validation MAE (unscaled)')
    plt.title(f'{run_tag} - Model MAE (unscaled)')
    plt.xlabel('Epochs');
    plt.ylabel('Mean Absolute Error');
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(dirs["plots_dir"], "training_curves.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"**Display: Training curves plot saved to {plot_filename}**")


def main(args):
    # --- Setup ---
    dataset_info = DATASET_OPTIONS[args.dataset_type]
    data_path = dataset_info["path"]
    dataset_name = dataset_info["name"]
    print(f"**Display: Running Experiment '{args.experiment}' with dataset: {dataset_name} from {data_path}**")

    device = setup_device()
    dirs = create_dirs(args.model_type, dataset_name, args.experiment)
    run_tag = dirs["run_tag"]  # Get the unique tag for this run

    # --- Data ---
    train_loader, val_loader, test_loader, scaler = build_exp_dataloaders(
        data_path, args.dataset_type, args.experiment, args.batch_size, RANDOM_STATE, IMG_X, IMG_Y, args.n_frames
    )

    # --- Model, Optimizer, Loss ---
    writer = SummaryWriter(log_dir=dirs["logs_dir"])
    model = get_model(args.model_type, args.n_frames, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    epoch_train_losses, epoch_val_losses, epoch_train_errors, epoch_val_errors = [], [], [], []
    print(f"\n**Display: Starting training for run '{run_tag}' for {args.epochs} epochs...**")
    for epoch in range(args.epochs):
        train_loss, train_error = train_epoch(model, device, train_loader, optimizer, criterion, epoch, scaler,
                                              args.log_interval)
        val_loss, val_error = validate_epoch(model, device, val_loader, criterion, scaler, epoch)

        epoch_train_losses.append(train_loss);
        epoch_val_losses.append(val_loss)
        epoch_train_errors.append(train_error);
        epoch_val_errors.append(val_error)
        writer.add_scalar('Loss/Train_RMSE', train_loss, epoch)
        writer.add_scalar('Loss/Validation_RMSE', val_loss, epoch)
        writer.add_scalar('Error/Train_MAE_Unscaled', train_error, epoch)
        writer.add_scalar('Error/Validation_MAE_Unscaled', val_error, epoch)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pth"
            checkpoint_save_path = os.path.join(dirs["checkpoint_dir"], checkpoint_name)
            torch.save(model.state_dict(), checkpoint_save_path)
            print(f"**Display: Saved checkpoint: {checkpoint_save_path}**")

        df_logs = pd.DataFrame({'train_rmse_loss': epoch_train_losses, 'val_rmse_loss': epoch_val_losses,
                                'train_mae_unscaled': epoch_train_errors, 'val_mae_unscaled': epoch_val_errors})
        df_logs.to_csv(os.path.join(dirs["logs_dir"], "training_log.csv"), index_label="epoch")

    print("**Display: Training finished.**")
    writer.close()

    # --- Plotting and Testing ---
    plot_metrics(args.epochs, epoch_train_losses, epoch_val_losses, epoch_train_errors, epoch_val_errors, run_tag, dirs)

    if test_loader:
        final_model_path = os.path.join(dirs["checkpoint_dir"], f"checkpoint_epoch_{args.epochs}.pth")
        print(f"\n**Display: Proceeding to test the final model from epoch {args.epochs}...**")
        test_model_and_save_results(model, device, test_loader, scaler, run_tag, dirs, checkpoint_path=final_model_path)
    else:
        print("\n**Display: Test set is empty, skipping final testing phase.**")

    print(f"**Display: Process for run '{run_tag}' completed.**")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and Evaluate models for hardness estimation based on specific experimental setups.")

    # 新增实验选择参数
    parser.add_argument('--experiment', type=str, required=True, choices=['1', '2', '3_basic', '3_choco'],
                        help="Which experiment to run: "
                             "'1': Test on hardness 17. "
                             "'2': Test on CYLIND shape. "
                             "'3_basic': Test on BASIC shapes. "
                             "'3_choco': Test on CHOCO shapes.")

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['lstm', 'gru', 'transformer', 'tcn', 'video_tf'],
                        help="Type of decoder/model to use.")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['dataset1', 'dataset2'],
                        help="Which dataset to use (dataset1: hardness5, dataset2: hardness1Data)")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f"Number of training epochs (default: {EPOCHS}).")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE}).")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE}).")
    parser.add_argument('--log_interval', type=int, default=LOG_INTERVAL,
                        help=f"Log progress every N batches (default: {LOG_INTERVAL}).")
    parser.add_argument('--n_frames', type=int, default=SEQ_LEN,
                        help=f"Number of frames per sample (default: {SEQ_LEN}).")

    parsed_args = parser.parse_args()

    # 更新全局变量以匹配命令行参数
    EPOCHS = parsed_args.epochs
    BATCH_SIZE = parsed_args.batch_size
    LEARNING_RATE = parsed_args.learning_rate
    LOG_INTERVAL = parsed_args.log_interval
    SEQ_LEN = parsed_args.n_frames

    main(parsed_args)

