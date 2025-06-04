
import os
# os.environ['PYTORCH_ENABLE_MPS_F16_CONV'] = '0' # 如果你的PyTorch版本较老，这个可能需要
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' # 尝试减少内存占用，有时有帮助

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functions import Model, EncoderCNN, DecoderRNN, Dataset_CRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.autograd.set_detect_anomaly(True)     # 有梯度为 NaN / inf 时直接报栈
torch.backends.cuda.enable_flash_sdp(False) # 即使在 mps 也加上，避免误用 fused dropout
torch.backends.mps.deterministic = True     # 避免 kernel 随机选算法
os.environ['PYTORCH_ENABLE_MPS_F16_CONV'] = '0'      # 关闭 fp16 卷积
# torch.backends.mps.allow_fp16_reduced_precision_reduction(False)  # PyTorch≥2.1


# Paths
# data_path = "D:/Pytorch 实战/Gelsight 硬度/example/hardness_5_2024_03_03"
data_path = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness_5_2024_03_03"

save_model_path = "./CRNN_ckpt0/"
os.makedirs(save_model_path, exist_ok=True)

# Architecture Params
CNN_fc_hidden1, CNN_fc_hidden2 = 512, 348
CNN_embed_dim = 256
img_x, img_y = 224, 224
dropout_p = 0.1
RNN_hidden_layers = 1
RNN_hidden_nodes = 256
RNN_FC_dim = 128

# Training Params
k = 5
epochs = 30
batch_size = 16
learning_rate = 1e-5
log_interval = 10

# Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("mps" if torch.backends.mps.is_available() else "cpu")

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': False}

# Data
all_Y_list = []
# fnames = os.listdir(data_path)
fnames = sorted([f for f in os.listdir(data_path) if not f.startswith('.')])
all_X_list = []
for f in fnames:
    # print(f)
    if f == '.DS_Store':
        continue
    label = int(f[7:9])
    all_Y_list.append(label)
    all_X_list.append(f)

train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_Y_list, test_size=0.2, random_state=42)
transform = transforms.Compose([transforms.Resize([img_x, img_y]), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

train_label = np.array(train_label, dtype=np.float32).reshape(-1, 1)
test_label = np.array(test_label, dtype=np.float32).reshape(-1, 1)

# scaler_train = StandardScaler()
# scaler_test = StandardScaler()
scaler = StandardScaler()
# train_label_scaled = scaler_train.fit_transform(train_label)
# test_label_scaled = scaler_test.fit_transform(test_label)
# train_mean = scaler_train.mean_.astype(np.float32)
# train_sigma = scaler_train.scale_.astype(np.float32)
# test_mean = scaler_test.mean_.astype(np.float32)
# test_sigma = scaler_test.scale_.astype(np.float32)
train_label_scaled = scaler.fit_transform(train_label)
test_label_scaled = scaler.transform(test_label)
train_mean = scaler.mean_.astype(np.float32)
train_sigma = scaler.scale_.astype(np.float32)
test_mean = scaler.mean_.astype(np.float32)
test_sigma = scaler.scale_.astype(np.float32)

train_set = Dataset_CRNN(data_path, train_list, train_label_scaled, train_label, transform=transform)
valid_set = Dataset_CRNN(data_path, test_list, test_label_scaled, test_label, transform=transform)
train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device, dtype=torch.float32)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers,
                         h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to('cpu')
model = Model(cnn_encoder, rnn_decoder)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def train(model, device, train_loader, optimizer, epoch, mean_sigma):
    model.train()
    mean, sigma = mean_sigma
    MSE_Loss = nn.MSELoss()
    losses = []
    error = []
    N_count = 0
    for batch_idx, (X, y_scale, y) in enumerate(train_loader):
        X, y_scale, y = X.to(device, dtype=torch.float32), y_scale.to(device, dtype=torch.float32), y.numpy()
        optimizer.zero_grad()
        output = model(X)
        loss = torch.sqrt(MSE_Loss(output, y_scale))
        # loss = MSE_Loss(output, y_scale)
        losses.append(loss.item())
        loss.backward()
        # ---- 打印梯度信息 ----
        # if batch_idx % log_interval == 0:  # 每隔几个batch打印一次
        #     print(f"\n--- Grad Info (Epoch {epoch + 1}, Batch {batch_idx}, Device {device}) ---")
        #     total_grad_norm = 0
        #     max_grad_val = 0
        #     min_grad_val = float('inf')
        #     num_none_grads = 0
        #     num_zero_grads = 0
        #     num_params = 0
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             grad_norm = param.grad.norm().item()
        #             total_grad_norm += grad_norm
        #             max_grad_val = max(max_grad_val, param.grad.abs().max().item())
        #             min_grad_val = min(min_grad_val, param.grad.abs().min().item())
        #             if torch.all(param.grad == 0):
        #                 num_zero_grads += 1
        #             # print(f"Param: {name}, Grad Norm: {grad_norm:.4e}, Grad Mean Abs: {param.grad.abs().mean().item():.4e}")
        #         else:
        #             # print(f"Param: {name}, Grad: None")
        #             num_none_grads += 1
        #         num_params += 1
        #     print(f"Total Grad Norm: {total_grad_norm:.4e}")
        #     print(f"Max Grad Abs Value: {max_grad_val:.4e}")
        #     print(f"Min Grad Abs Value (non-zero): {min_grad_val:.4e} (if inf, all grads might be zero or None)")
        #     print(f"Params with None grad: {num_none_grads}/{num_params}")
        #     print(f"Params with all-zero grad: {num_zero_grads}/{num_params - num_none_grads}")
        # ---- 梯度信息结束 ----
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 或者尝试0.5, 5.0等值
        optimizer.step()
        pred = (output.detach().cpu().numpy() * sigma + mean)
        distance = np.mean((np.abs(pred - y) / y).squeeze())
        error.append(distance)
        N_count += X.size(0)
        # print for every log_interval batches
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\trmse_Loss: {:.6f}, avg_error: {:.2f}'.format(
                epoch + 1, N_count, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), 100. * distance))
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\trmse_Loss: {:.6f}, avg_error: {:.2f}'.format(
        #     epoch + 1, N_count, len(train_loader.dataset),
        #     100. * (batch_idx + 1) / len(train_loader), loss.item(), 100. * distance))
    return np.mean(losses), np.mean(error)

def validation(model, device, optimizer, test_loader, mean_sigma):
    model.eval()
    mean, sigma = mean_sigma
    MSE_Loss = nn.MSELoss()
    test_loss = 0
    error = []
    pred_list = []
    true_label = []
    i = 0
    with torch.no_grad():
        for X, y_scale, y in test_loader:
            X, y_scale, y = X.to(device), y_scale.to(device), y.numpy()
            output = model(X)
            loss = torch.sqrt(MSE_Loss(output, y_scale))
            pred = (output.detach().cpu().numpy() * sigma + mean)
            distance = np.abs(pred - y).squeeze() / y.squeeze()
            if distance.ndim == 0:
                distance = np.array([distance])
            error.extend(distance.tolist())
            pred_list.append(pred)
            true_label.append(y)
            test_loss += loss.item()
            i += 1
    test_loss /= i
    pred_list = np.concatenate(pred_list, axis=0).squeeze()
    true_label = np.concatenate(true_label, axis=0).squeeze()
    avg_error = np.mean(error)
    print(f"\nTest set ({len(test_loader.dataset)} samples): rmse_loss: {test_loss:.4f}, avg_error: {100 * avg_error:.2f}\n")
    torch.save(model.state_dict(), os.path.join(save_model_path, f"model{epoch + 1}.pth"))
    return test_loss, avg_error, pred_list, true_label

# Logs
epoch_train_losses, epoch_train_error = [], []
epoch_test_losses, epoch_test_error = [], []
epoch_pred_list, epoch_true_label = [], []

# Train loop
for epoch in range(epochs):
    print(f"Running on device: {device}")
    train_loss, train_error = train(model, device, train_loader, optimizer, epoch, mean_sigma=[train_mean, train_sigma])
    test_loss, test_error, pred_list, true_label = validation(model, device, optimizer, valid_loader, mean_sigma=[test_mean, test_sigma])
    epoch_train_losses.append(train_loss)
    epoch_train_error.append(train_error)
    epoch_test_losses.append(test_loss)
    epoch_test_error.append(test_error)
    epoch_pred_list.append(pred_list)
    epoch_true_label.append(true_label)
    np.save('./CRNN_epoch_training_losses.npy', np.array(epoch_train_losses))
    np.save('./CRNN_epoch_training_error.npy', np.array(epoch_train_error))
    np.save('./CRNN_epoch_test_loss.npy', np.array(epoch_test_losses))
    np.save('./CRNN_epoch_test_error.npy', np.array(epoch_test_error))
    np.save('./CRNN_epoch_pred_list.npy', np.array(epoch_pred_list))
    np.save('./CRNN_epoch_true_label.npy', np.array(epoch_true_label))

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), epoch_train_losses)
plt.plot(np.arange(1, epochs + 1), epoch_test_losses)
plt.title("model loss")
plt.xlabel("epochs")
plt.ylabel("rmse_loss")
plt.legend(["train", "test"])
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), epoch_train_error)
plt.plot(np.arange(1, epochs + 1), epoch_test_error)
plt.title("regression error")
plt.xlabel("epochs")
plt.ylabel("error")
plt.legend(["train", "test"])
plt.savefig("./fig_hardness5_CRNN.png", dpi=600)
# plt.show()
