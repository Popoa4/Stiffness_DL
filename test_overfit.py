import os, torch, numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from functions import Dataset_CRNN, VideoTransformer           # <-- 保证能 import

# ---------- 基本参数 ----------
DATA_PATH      = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness_5_2024_03_03"
DATASET_TYPE   = "dataset1"              # 'dataset1' / 'dataset2'
IDX_START      = 0                       # 选哪条样本
N_FRAMES       = 16
IMG_SIZE       = 224
BATCH_SIZE     = 8                       # 1 条样本反复循环
EPOCHS         = 500
LR             = 1e-3                    # 夸张一点
EMBED_DIM      = 128
DEPTH          = 2
N_HEAD         = 2
PATCH_SIZE     = 16

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"[info] device = {device}")

# ---------- 读取所有文件名 ----------
fnames = sorted([f for f in os.listdir(DATA_PATH)
                 if os.path.isdir(os.path.join(DATA_PATH, f))])

lab_s = 7 if DATASET_TYPE == 'dataset1' else 8
lab_e = lab_s + 2

labels, folders = [], []
for f in fnames:
    try:
        labels.append(float(f[lab_s:lab_e]))
        folders.append(f)
    except Exception: pass

labels = np.array(labels, dtype=np.float32).reshape(-1,1)

# ---------- 只取一条样本 ----------
idx = [IDX_START]
print(f"[info] pick sample folder = {folders[idx[0]]}, label = {labels[idx[0],0]}")

transform = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.ToTensor()                                   # 不做归一化
])

ds_full  = Dataset_CRNN(DATA_PATH, folders, labels, labels,
                        n_frames=N_FRAMES, transform=transform)
one_ds   = Subset(ds_full, idx)
loader   = DataLoader(one_ds, batch_size=BATCH_SIZE, shuffle=True)

# ---------- 构造模型 ----------
model = VideoTransformer(img_size=IMG_SIZE,
                         patch_size=PATCH_SIZE,
                         n_frames=N_FRAMES,
                         embed_dim=EMBED_DIM,
                         depth=DEPTH,
                         n_head=N_HEAD,
                         mlp_ratio=2.,
                         drop_p=0.1).to(device)

# 可选：冻结早期层，只训练 head
# for name, p in model.named_parameters():
#     if name.startswith("temporal_transformer.layers.1") or \
#             name.startswith("head"):
#         p.requires_grad = True
#     else:
#         p.requires_grad = False

optimiser = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                        lr=LR)
loss_fn   = nn.MSELoss()

# ---------- 训练 ----------
for epoch in range(1, EPOCHS+1):
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimiser.step()

    if epoch % 20 == 0 or epoch == 1:
        rmse = torch.sqrt(loss).item()
        print(f"Epoch {epoch:4d}/{EPOCHS}, MSE={loss.item():.6f}, RMSE={rmse:.6f}")

    # 提前退出：足够小就停
    if torch.sqrt(loss).item() < 1e-2:
        print(f"Converged at epoch {epoch}")
        break
