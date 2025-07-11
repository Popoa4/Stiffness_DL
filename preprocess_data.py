# preprocess_data.py
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

# 假设你的主代码中的解析函数在这里也能用
from main import parse_dir_info, DATASET_OPTIONS, SHAPES_ALL

data_path = './Dataset/hardnepreprocessed_datass-1Data'
save_path = './Dataset/'

def preprocess_and_save(data_path, save_path, n_frames, img_size):
    """
    读取原始图片数据，处理后保存为.pt文件
    """
    os.makedirs(save_path, exist_ok=True)

    # 使用和训练时完全一致的图像变换
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"开始预处理数据，从 {data_path} 保存到 {save_path}")

    # 遍历所有样本文件夹
    all_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    for dirname in tqdm(all_dirs):
        folder_path = os.path.join(data_path, dirname)

        # ----------- 复用 Dataset_CRNN.read_images 的核心逻辑 -----------
        frame_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))])
        frame_count = len(frame_files)

        if frame_count == 0:
            print(f"警告：文件夹 {dirname} 为空，已跳过。")
            continue

        if frame_count >= n_frames:
            idx_array = np.linspace(0, frame_count - 1, num=n_frames, dtype=int)
        else:
            idx_array = np.concatenate([
                np.arange(frame_count),
                np.full(n_frames - frame_count, frame_count - 1)
            ]).astype(int)

        imgs = []
        for idx in idx_array:
            img_path = os.path.join(folder_path, frame_files[idx])
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                imgs.append(img_tensor)
            except Exception as e:
                print(f"警告：读取 {img_path} 失败: {e}，跳过此样本。")
                break  # 如果一个样本的帧有问题，最好跳过整个样本

        if len(imgs) != n_frames:
            continue
        # -----------------------------------------------------------------

        # 堆叠成一个Tensor
        data_tensor = torch.stack(imgs, dim=0)  # Shape: [n_frames, C, H, W]

        # 获取标签
        try:
            # 假设你使用的是 dataset2
            shape, hardness = parse_dir_info(dirname, 'dataset2')
            label_value = float(hardness)
        except ValueError:
            print(f"警告：无法解析文件夹名称 {dirname}，已跳过。")
            continue

        # 将数据和标签保存在一个字典中，然后保存为.pt文件
        # 保存为 .pt 比 .npy 更好，因为torch.load更快
        save_file_path = os.path.join(save_path, f"{dirname}.pt")
        torch.save({
            'data': data_tensor,
            'label': label_value  # 直接保存原始硬度值
        }, save_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument('--data_path', type=str, required=False, help="原始数据文件夹路径")
    parser.add_argument('--save_path', type=str, required=False, help="预处理后数据的保存路径")
    parser.add_argument('--n_frames', type=int, default=10, help="每个样本的帧数")
    parser.add_argument('--img_size', type=int, default=224, help="图像大小")
    args = parser.parse_args()

    preprocess_and_save(data_path, save_path, args.n_frames, args.img_size)
    print("预处理完成！")
