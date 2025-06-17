import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# 确保能从 functions.py 导入所需模块
try:
    from functions import Dataset_CRNN, VideoTransformer
except ImportError:
    print("错误：请确保 `functions.py` 文件与此脚本在同一目录下。")
    exit()

# ===================================================================
# ---              配置参数 (可根据需要修改)                      ---
# ===================================================================
# **Display: 1. 请务必修改为您的数据集路径**
DATA_PATH = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness_5_2024_03_03"
# **Display: 2. 选择数据集格式 ('dataset1' or 'dataset2')**
DATASET_TYPE = "dataset1"

# **Display: 3. 过拟合测试的核心参数**
NUM_SAMPLES_TO_OVERFIT = 8  # **用多少个样本进行过拟合测试**
BATCH_SIZE = 4  # **批大小 (建议为 NUM_SAMPLES 的因子)**
LEARNING_RATE = 3e-4  # **为Transformer从头训练推荐的较大学习率**
EPOCHS = 200  # **足够多的轮次以观察损失下降**
N_FRAMES = 16  # **与主脚本保持一致的帧数**
IMG_SIZE = 224

# **Display: 4. 模型参数 (与主脚本保持一致)**
MODEL_EMBED_DIM = 384
MODEL_DEPTH = 6
MODEL_N_HEAD = 6
MODEL_PATCH_SIZE = 16


# ===================================================================

def run_overfit_test():
    """主函数，执行过拟合测试"""

    # --- 1. 环境设置 ---
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"**Display: 使用设备: {device}**")
    print(f"**Display: 过拟合测试启动，目标：用 {NUM_SAMPLES_TO_OVERFIT} 个样本将训练损失降至接近零。**\n")

    # --- 2. 数据加载与子集创建 ---
    print("**Display: 正在加载完整数据集以创建标准化器(Scaler)和小子集...**")

    # 简化版的数据加载流程，与主脚本逻辑一致
    all_Y_list_raw = []
    all_X_list_fnames = []
    fnames = sorted(
        [f for f in os.listdir(DATA_PATH) if not f.startswith('.') and os.path.isdir(os.path.join(DATA_PATH, f))])

    label_start_idx = 7 if DATASET_TYPE == 'dataset1' else 8
    label_end_idx = label_start_idx + 2

    for f_dir in fnames:
        try:
            label = int(f_dir[label_start_idx:label_end_idx])
            all_Y_list_raw.append(label)
            all_X_list_fnames.append(f_dir)
        except (ValueError, IndexError):
            continue

    if not all_X_list_fnames:
        raise ValueError(f"在路径 {DATA_PATH} 中找不到有效数据。")

    all_Y_list_raw = np.array(all_Y_list_raw, dtype=np.float32).reshape(-1, 1)

    # 仅使用训练集部分来创建scaler和我们的过拟合小子集
    X_train, _, Y_train_raw, _ = train_test_split(
        all_X_list_fnames, all_Y_list_raw, test_size=0.40, random_state=42  # 划分出60%作为“训练”
    )

    # 创建并拟合标准化器
    scaler = StandardScaler()
    Y_train_scaled = scaler.fit_transform(Y_train_raw)

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize([IMG_SIZE, IMG_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建完整的训练数据集对象
    full_train_dataset = Dataset_CRNN(DATA_PATH, X_train, Y_train_scaled, Y_train_raw, transform=transform,
                                      n_frames=N_FRAMES)

    # **核心步骤：从完整训练集中抽取一个小子集用于过拟合**
    if len(full_train_dataset) < NUM_SAMPLES_TO_OVERFIT:
        raise ValueError(
            f"请求的过拟合样本数({NUM_SAMPLES_TO_OVERFIT}) 大于可用训练样本总数({len(full_train_dataset)})")

    overfit_subset_indices = list(range(NUM_SAMPLES_TO_OVERFIT))
    overfit_dataset = Subset(full_train_dataset, overfit_subset_indices)

    # 创建数据加载器
    overfit_loader = DataLoader(overfit_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"**Display: 已成功创建包含 {len(overfit_dataset)} 个样本的过拟合数据集。**\n")

    # --- 3. 模型、优化器、损失函数 ---
    print(
        f"**Display: 初始化VideoTransformer模型 (dim={MODEL_EMBED_DIM}, depth={MODEL_DEPTH}, heads={MODEL_N_HEAD})...**")
    model = VideoTransformer(
        img_size=IMG_SIZE,
        patch_size=MODEL_PATCH_SIZE,
        n_frames=N_FRAMES,
        embed_dim=MODEL_EMBED_DIM,
        depth=MODEL_DEPTH,
        n_head=MODEL_N_HEAD,
        mlp_ratio=2.0,
        drop_p=0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses = []

    # --- 4. 训练循环 ---
    print("**Display: 开始训练...**")
    model.train()

    pbar = tqdm(range(EPOCHS), desc="Overfitting Progress")
    for epoch in pbar:
        epoch_loss = 0.0
        num_batches = 0

        for X, y_scaled, _ in overfit_loader:
            X, y_scaled = X.to(device), y_scaled.to(device)

            optimizer.zero_grad()

            output = model(X)

            loss = criterion(output, y_scaled)
            rmse_loss = torch.sqrt(loss)

            loss.backward()
            optimizer.step()

            epoch_loss += rmse_loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)

        pbar.set_postfix_str(f"Epoch {epoch + 1}/{EPOCHS}, RMSE Loss: {avg_epoch_loss:.6f}")

    print("\n**Display: 训练完成。**")

    # --- 5. 结果分析与可视化 ---
    final_loss = train_losses[-1]
    print(f"**Display: 最终训练RMSE Loss: {final_loss:.6f}**")

    if final_loss < 0.1:
        print("✅ **Display: 成功！模型已在小子集上过拟合，损失显著下降。这表明模型和训练流程基本工作正常。**")
        print(
            "   **下一步建议**：可以回到主脚本，尝试在完整数据集上使用更高的学习率(如1e-4)、调整模型大小或使用预训练权重。")
    else:
        print("❌ **Display: 失败。模型未能在小子集上过拟合。**")
        print("   **可能原因**：")
        print("   1. **模型对于此任务过于复杂或不合适**：尝试一个更小的VideoTransformer（降低embed_dim, depth, n_head）。")
        print("   2. **初始化问题**：随机初始化可能陷入了糟糕的局部最小值。")
        print("   3. **数据本身问题**：检查前几帧图像是否内容相似度过高或信息量过低。")
        print("   **下一步建议**：首先大幅简化模型（例如 `depth=2`, `embed_dim=128`）再运行此测试。")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', linestyle='-')
    plt.title("Overfitting Test - Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Loss")
    plt.grid(True)
    plt.yscale('log')  # 使用对数坐标轴以便观察早期变化
    plt.show()


if __name__ == '__main__':
    run_overfit_test()
