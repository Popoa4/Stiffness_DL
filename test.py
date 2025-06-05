# --------------------------------------------------
# test.py  ―― 评估已训练好的 model30_LSTM.pth
# --------------------------------------------------
import os, math, numpy as np, torch
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from functions import (Model, EncoderCNN, DecoderRNN,
                       Dataset_CRNN, DecoderTCN, DecoderGRU, DecoderTransformer)          # 与训练时保持一致

# ----------------- 环境与基本参数 -----------------
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(">>> Device :", device)

data_path  = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness_5_2024_03_03"
ckpt_path  = "./model30_TCN.pth"          # 模型文件
assert os.path.exists(ckpt_path), f"模型文件 {ckpt_path} 不存在!"

# 与训练脚本一致的网络 & 训练参数
CNN_fc_hidden1, CNN_fc_hidden2 = 512, 348
CNN_embed_dim   = 256
img_x, img_y    = 224, 224
dropout_p       = 0.1
RNN_hidden_layers = 1
RNN_hidden_nodes  = 256
RNN_FC_dim        = 128
k                 = 5                     # 训练脚本里传入的 num_classes
batch_size        = 16

# ----------------- 构建数据集 -----------------
# 读取文件名并解析标签
fnames = sorted([f for f in os.listdir(data_path) if not f.startswith('.')])
X_all, y_all = [], []
for f in fnames:
    try:
        y_all.append(int(f[7:9]))   # 与训练脚本相同的解析方式
        X_all.append(f)
    except:
        continue

# 保证与训练时同样的划分
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42)

# 标签标准化 (同一个 StandardScaler)
scaler = StandardScaler().fit(np.array(y_train, dtype=np.float32).reshape(-1, 1))
y_train_scaled = scaler.transform(np.array(y_train, dtype=np.float32).reshape(-1, 1))
y_test_scaled  = scaler.transform(np.array(y_test , dtype=np.float32).reshape(-1, 1))
mean_, sigma_  = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)

# 图像变换
transform = transforms.Compose([
    transforms.Resize([img_x, img_y]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 只需要测试集
test_set   = Dataset_CRNN(data_path, X_test, y_test_scaled, y_test,
                          transform=transform)
test_loader = data.DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False, num_workers=0)

# ----------------- 定义模型并加载权重 -----------------
cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y,
                         fc_hidden1=CNN_fc_hidden1,
                         fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout_p,
                         CNN_embed_dim=CNN_embed_dim).to(device, dtype=torch.float32)

# rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim,
#                          h_RNN_layers=RNN_hidden_layers,
#                          h_RNN=RNN_hidden_nodes,
#                          h_FC_dim=RNN_FC_dim,
#                          drop_p=dropout_p,
#                          num_classes=k).to('cpu', dtype=torch.float32)

# selection2: GRU
# rnn_decoder = DecoderGRU(CNN_embed_dim=CNN_embed_dim,
#                         h_RNN_layers=RNN_hidden_layers,
#                         h_RNN=RNN_hidden_nodes,
#                         h_FC_dim=RNN_FC_dim,
#                         drop_p=dropout_p,
#                         num_classes=k).to(device, dtype=torch.float32)
# selection3: Transformer
# rnn_decoder = DecoderTransformer(CNN_embed_dim=CNN_embed_dim,
#                                nhead=4,  # 可调整
#                                num_layers=2,  # 可调整
#                                h_FC_dim=RNN_FC_dim,
#                                drop_p=dropout_p,
#                                num_classes=k).to(device, dtype=torch.float32)
# selection4: TCN
rnn_decoder = DecoderTCN(CNN_embed_dim=CNN_embed_dim,
                        num_levels=3,  # 可调整，影响感受野大小
                        h_FC_dim=RNN_FC_dim,
                        drop_p=dropout_p,
                        num_classes=k).to(device, dtype=torch.float32)

model = Model(cnn_encoder, rnn_decoder)

state = torch.load(ckpt_path, map_location=device, weights_only=True)
# 支持直接 state_dict 或 {'state_dict': ...}
model.load_state_dict(state['state_dict'] if isinstance(state, dict) and
                      'state_dict' in state else state)
model.eval()
print(f">>> Loaded checkpoint: {ckpt_path}")

# ----------------- 推理与指标 -----------------
pred_scaled, true_scaled = [], []
pred_orig , true_orig    = [], []

with torch.no_grad():
    for X, y_s, y_orig in test_loader:
        X   = X.to(device, dtype=torch.float32)
        y_s = y_s.to(device, dtype=torch.float32)     # (B,1)

        outputs = model(X)                            # (B,1) 或 (B,k)
        if outputs.ndim == 2 and outputs.size(1) > 1:
            outputs = outputs[:, 0:1]                 # 只取第 0 通道

        # --- 把每个 batch 的结果摊平成 1-D，再 extend ---
        pred_scaled.extend(outputs.cpu().view(-1).numpy())
        true_scaled.extend(y_s.cpu().view(-1).numpy())

        outs_orig = outputs.cpu().numpy() * sigma_ + mean_
        pred_orig.extend(outs_orig.reshape(-1))
        true_orig.extend(y_orig.numpy().reshape(-1))

# 转成 numpy 向量 (N,)
pred_scaled = np.array(pred_scaled)
true_scaled = np.array(true_scaled)
pred_orig   = np.array(pred_orig)
true_orig   = np.array(true_orig)


# --------- 评价指标 ----------
rmse = math.sqrt(mean_squared_error(true_orig, pred_orig))
mae  = mean_absolute_error(true_orig, pred_orig)
r2   = r2_score(true_orig, pred_orig)
rel_error = np.mean(np.abs(pred_orig - true_orig) / true_orig)  # 与训练脚本中的 avg_error 对齐

print("\n========== Test  Results ==========")
print(f"RMSE                 : {rmse:.4f}")
print(f"MAE                  : {mae :.4f}")
print(f"R²                   : {r2  :.4f}")
print(f"Mean Relative Error  : {rel_error*100:.2f}%")
print("===================================\n")
# 保存预测结果
import pandas as pd
df = pd.DataFrame({
    'filename': X_test,
    'true_value': true_orig,
    'pred_value': pred_orig
})
df.to_csv("test_predictions_LSTM.csv", index=False)