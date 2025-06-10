import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from functions.py
from functions import (
    Dataset_CRNN, EncoderCNN, DecoderRNN, DecoderGRU,
    DecoderTransformer, DecoderTCN, Model
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
DROPOUT_P = 0.1
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
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOG_INTERVAL = 10  # Print training log every N batches
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
def get_data_loaders(data_path, batch_size, random_state, img_x, img_y, dataset_type):
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

    train_dataset = Dataset_CRNN(data_path, X_train, Y_train_scaled, Y_train_raw, transform=transform)
    val_dataset = Dataset_CRNN(data_path, X_val, Y_val_scaled, Y_val_raw, transform=transform)
    test_dataset = Dataset_CRNN(data_path, X_test, Y_test_scaled, Y_test_raw, transform=transform)

    # Dataloader params from original code
    loader_params = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': False}  # pin_memory True if using CUDA

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)

    return train_loader, val_loader, test_loader, scaler


def get_model(model_type: str, device):
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
    total_loss = 0.0
    total_abs_err_unscaled = 0.0
    total_samples = 0

    # Get device of the first parameter of the decoder for target y_scale device
    # For LSTM, this will be CPU. For others, it will be the main device.
    decoder_param_device = next(model.decoder.parameters()).device

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
    for batch_idx, (X, y_scale, y_orig) in pbar:
        X = X.to(device, dtype=torch.float32)  # Input to CNN encoder
        y_scale_target = y_scale.to(decoder_param_device, dtype=torch.float32)  # Target on decoder's device

        optimizer.zero_grad()
        output = model(X)  # Output will be on decoder_param_device

        loss = criterion(output, y_scale_target)
        rmse_loss = torch.sqrt(loss)  # Original code uses sqrt of MSE
        rmse_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += rmse_loss.item() * X.size(0)

        # For error calculation on original scale
        pred_unscaled = scaler.inverse_transform(output.detach().cpu().numpy())
        abs_error_unscaled = np.abs(pred_unscaled - y_orig.numpy())
        total_abs_err_unscaled += np.sum(abs_error_unscaled)
        total_samples += X.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss_so_far = total_loss / total_samples
            avg_mae_unscaled_so_far = total_abs_err_unscaled / total_samples
            pbar.set_postfix_str(f"RMSE Loss: {avg_loss_so_far:.4f}, MAE (unscaled): {avg_mae_unscaled_so_far:.2f}")

    avg_epoch_loss = total_loss / total_samples
    avg_epoch_mae_unscaled = total_abs_err_unscaled / total_samples
    return avg_epoch_loss, avg_epoch_mae_unscaled


def validate_epoch(model, device, val_loader, criterion, scaler, epoch=None):
    model.eval()
    total_loss = 0.0
    total_abs_err_unscaled = 0.0
    total_samples = 0

    decoder_param_device = next(model.decoder.parameters()).device

    desc_str = "Validation"
    if epoch is not None: desc_str = f"Epoch {epoch + 1}/{EPOCHS} [Val]"

    pbar = tqdm(val_loader, total=len(val_loader), desc=desc_str)
    with torch.no_grad():
        for X, y_scale, y_orig in pbar:
            X = X.to(device, dtype=torch.float32)
            y_scale_target = y_scale.to(decoder_param_device, dtype=torch.float32)

            output = model(X)
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
    print(f"{desc_str} Results: Avg RMSE Loss: {avg_epoch_loss:.4f}, Avg MAE (unscaled): {avg_epoch_mae_unscaled:.2f}")
    return avg_epoch_loss, avg_epoch_mae_unscaled


def test_model_and_save_results(model, device, test_loader, scaler, model_name, dataset_name, dirs,
                                checkpoint_path=None):
    if checkpoint_path:
        print(f"**Display: Loading model from {checkpoint_path} for testing.**")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except RuntimeError as e:
            print(f"Standard load failed ({e}), attempting careful load for mixed devices (e.g. LSTM)...")
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            # Ensure modules are on their designated devices after loading
            model.EncoderCNN.to(device)
            if model.model_type == 'lstm':
                model.decoder.to(torch.device('cpu'))
            else:
                model.decoder.to(device)
        print(f"**Display: Model loaded from {checkpoint_path} successfully.**")

    model.eval()
    all_predictions_unscaled = []
    all_true_values_unscaled = []
    all_scaled_predictions = []
    all_scaled_true = []

    decoder_param_device = next(model.decoder.parameters()).device

    print("**Display: Starting testing phase...**")
    pbar = tqdm(test_loader, total=len(test_loader), desc="Testing")
    with torch.no_grad():
        for X, y_scale, y_orig in pbar:
            X = X.to(device, dtype=torch.float32)
            output_scaled = model(X)  # Scaled output from model

            pred_unscaled = scaler.inverse_transform(output_scaled.detach().cpu().numpy())

            all_predictions_unscaled.extend(pred_unscaled.flatten().tolist())
            all_true_values_unscaled.extend(y_orig.numpy().flatten().tolist())
            all_scaled_predictions.extend(output_scaled.detach().cpu().numpy().flatten().tolist())
            all_scaled_true.extend(y_scale.numpy().flatten().tolist())

    # Calculate metrics
    true_np = np.array(all_true_values_unscaled)
    pred_np = np.array(all_predictions_unscaled)

    mse = mean_squared_error(true_np, pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_np, pred_np)
    r2 = r2_score(true_np, pred_np)

    # Relative error calculation
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

    # Save predictions
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
        data_path, args.batch_size, RANDOM_STATE, IMG_X, IMG_Y, args.dataset_type
    )

    # Model
    model = get_model(args.model_type, device)

    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()  # We will take sqrt for RMSE as in original code

    # Training Loop
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_errors, epoch_val_errors = [], []  # Using MAE (unscaled) for error tracking

    print(
        f"\n**Display: Starting training for {args.model_type.upper()} model on {dataset_name} for {args.epochs} epochs...**")
    for epoch in range(args.epochs):
        train_loss, train_error = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch, scaler, args.log_interval
        )
        val_loss, val_error = validate_epoch(
            model, device, val_loader, criterion, scaler, epoch
        )

        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_train_errors.append(train_error)
        epoch_val_errors.append(val_error)

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
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'gru', 'transformer', 'tcn'],
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

    parsed_args = parser.parse_args()

    # Ensure globals are updated
    EPOCHS = parsed_args.epochs
    BATCH_SIZE = parsed_args.batch_size
    LEARNING_RATE = parsed_args.learning_rate
    LOG_INTERVAL = parsed_args.log_interval

    main(parsed_args)
