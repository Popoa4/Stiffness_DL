import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import math

## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


## ---------------------- Dataloaders ---------------------- ##
class Dataset_CRNN(data.Dataset):
    def __init__(self, data_path, folders, labels, gt, n_frames = 10, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.gt_labels = gt
        self.folders = folders
        self.transform = transform
        self.n_frames = n_frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        """
        始终返回 self.n_frames 张经过 transform 的图片：
        1. 帧够用   → 均匀抽 self.n_frames 张
        2. 帧不足   → 先取所有帧，再重复最后一帧补齐
        3. 路径缺失 → 回退到最后一张成功读取的帧
        """
        folder_path = os.path.join(path, selected_folder)
        # 只保留真正的图片文件，避免 .DS_Store 或其它杂项
        frame_files = sorted([f for f in os.listdir(folder_path)
                              if f.lower().endswith(('.jpg', '.png'))])
        frame_count = len(frame_files)
        wanted = self.n_frames

        if frame_count == 0:
            raise RuntimeError(f'Folder {folder_path} 没有任何帧！')

        # -------- 1) 生成要读取的帧索引 --------
        if frame_count >= wanted:  # 帧数足够
            # linspace 比 arange + skip_frame 更均匀也更直观
            idx_array = np.linspace(0, frame_count - 1,
                                    num=wanted, dtype=int)
        else:  # 帧数不够
            print(f'[Dataset Warning] {selected_folder} 只有 {frame_count} 帧，'
                  f'需要 {wanted} 帧，自动复制最后一帧补齐。')
            # 先拿到全部现有帧，再把最后一帧重复若干次
            idx_array = np.concatenate([
                np.arange(frame_count),
                np.full(wanted - frame_count, frame_count - 1)
            ]).astype(int)

        # -------- 2) 读取并 transform --------
        imgs = []
        last_valid_img = None
        for idx in idx_array:
            # 试着用统一命名规则 `frame_xxxx.jpg`
            img_path = os.path.join(folder_path, f'frame_{idx:04d}.jpg')
            if not os.path.exists(img_path):  # 文件名不规范，退回列表索引
                if idx < frame_count:
                    img_path = os.path.join(folder_path, frame_files[idx])
                else:  # 极少发生；用最后一张兜底
                    img_path = os.path.join(folder_path, frame_files[-1])

            try:
                img = Image.open(img_path).convert('RGB')
                last_valid_img = img
            except Exception as e:  # 读不到就复制上一张
                print(f'[Dataset Warning] 读取 {img_path} 失败：{e}')
                if last_valid_img is None:
                    # 如果第一张就失败，造一张黑图
                    img = Image.new('RGB', (self.transform_size, self.transform_size))
                else:
                    img = last_valid_img

            if use_transform is not None:
                img = use_transform(img)
            imgs.append(img)

        # -------- 3) 拼成 [wanted, C, H, W] --------
        return torch.stack(imgs, dim=0)  # Tensor

    def __getitem__(self, index):
        folder = self.folders[index]
        X = self.read_images(self.data_path, folder, self.transform)
        Y_scale = torch.tensor(self.labels[index], dtype=torch.float)
        Y = self.gt_labels[index]
        return X, Y_scale, Y


## -------------------- CRNN prediction ---------------------- ##
def CRNN_final_prediction(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            X = X.to(device)
            output = rnn_decoder(cnn_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred

## ------------------------ CRNN module ---------------------- ##
class EncoderCNN(nn.Module):
    def __init__(self, img_x=90, img_y=120, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)

        def conv2D_output_size(img_size, padding, kernel_size, stride):
            return (
                np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int)
            )

        conv1_outshape = conv2D_output_size((img_x, img_y), self.pd1, self.k1, self.s1)
        conv2_outshape = conv2D_output_size(conv1_outshape, self.pd2, self.k2, self.s2)
        conv3_outshape = conv2D_output_size(conv2_outshape, self.pd3, self.k3, self.s3)
        conv4_outshape = conv2D_output_size(conv3_outshape, self.pd4, self.k4, self.s4)

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.ch2, self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.ch3, self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(self.ch4 * conv4_outshape[0] * conv4_outshape[1], self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            x = self.conv1(x_3d[:, t])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq

class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=256, h_RNN_layers=1, h_RNN=256, h_FC_dim=128, drop_p=0.3):
        super(DecoderRNN, self).__init__()
        self.LSTM = nn.LSTM(input_size=CNN_embed_dim, hidden_size=h_RNN, num_layers=h_RNN_layers, batch_first=True)
        self.fc1 = nn.Linear(h_RNN, h_FC_dim)
        self.fc2 = nn.Linear(h_FC_dim, 1)
        self.drop_p = drop_p

    def forward(self, x_RNN):
        RNN_out, _ = self.LSTM(x_RNN)
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x

## -------------------- DecoderGRU module ---------------------- ##
class DecoderGRU(nn.Module):
    def __init__(self, CNN_embed_dim=256, h_RNN_layers=1, h_RNN=256, h_FC_dim=128, drop_p=0.3):
        super(DecoderGRU, self).__init__()
        self.GRU = nn.GRU(input_size=CNN_embed_dim,
                          hidden_size=h_RNN,
                          num_layers=h_RNN_layers,
                          batch_first=True)
        self.fc1 = nn.Linear(h_RNN, h_FC_dim)
        self.fc2 = nn.Linear(h_FC_dim, 1)
        self.drop_p = drop_p

    def forward(self, x_RNN):
        RNN_out, _ = self.GRU(x_RNN)
        x = self.fc1(RNN_out[:, -1, :])  # 取最后一个时间步
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x

## ------------------- Transformer module ---------------------- ##
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, features]
        return x + self.pe[:, :x.size(1), :]


class DecoderTransformer(nn.Module):
    def __init__(self, CNN_embed_dim=256, nhead=4, num_layers=2, h_FC_dim=128, drop_p=0.3):
        super().__init__()
        # 位置编码
        self.pos_encoder = PositionalEncoding(CNN_embed_dim)

        # 创建Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CNN_embed_dim,
            nhead=nhead,
            dim_feedforward=CNN_embed_dim * 4,
            dropout=drop_p,
            batch_first=True  # 重要：保持[batch, seq, feature]顺序
        )

        # 堆叠编码器层
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出层
        self.fc1 = nn.Linear(CNN_embed_dim, h_FC_dim)
        self.fc2 = nn.Linear(h_FC_dim, 1)
        self.drop_p = drop_p

    def forward(self, x):
        # 添加位置编码
        x = self.pos_encoder(x)

        # 通过Transformer (不需要mask，因为我们只取最后输出)
        out = self.transformer(x)

        # 取序列最后一个时间步
        x = self.fc1(out[:, -1])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x

## -------------------- Temporal Convolutional Network (TCN) module ---------------------- ##
def Chomp1d(x, chomp_size):
    """裁掉多余的 time step（右侧裁剪）"""
    return x[:, :, :-chomp_size] if chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,
                 dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = lambda x: Chomp1d(x, padding)   # 关键
        self.bn1   = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = lambda x: Chomp1d(x, padding)   # 关键
        self.bn2   = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
                          if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)   # ↓↓↓ 裁剪后长度=L_in
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)   # ↓↓↓ 再裁剪一次
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class DecoderTCN(nn.Module):
    def __init__(self, CNN_embed_dim=256, num_levels=3, h_FC_dim=128, drop_p=0.3, kernel_size = 3):
        super().__init__()

        # TCN需要输入格式为 [batch, channels, seq_len]
        channels = [CNN_embed_dim] * num_levels  # 每层通道数
        # kernel_size = 3  # 卷积核大小

        # 构建TCN块
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i  # 指数级扩张感受野
            in_ch = CNN_embed_dim if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(
                in_ch, channels[i], kernel_size,
                stride=1,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation,  # 因果卷积需要的padding
                dropout=drop_p
            ))
        self.tcn = nn.Sequential(*layers)

        # 输出层
        self.fc1 = nn.Linear(channels[-1], h_FC_dim)
        self.fc2 = nn.Linear(h_FC_dim, 1)
        self.drop_p = drop_p

    def forward(self, x):
        # x: [batch, seq, features] -> [batch, features, seq]
        x = x.transpose(1, 2)

        # TCN处理
        x = self.tcn(x)

        # 取最后时间步的输出 [batch, channels, seq] -> [batch, channels]
        x = x[:, :, -1]

        # 全连接输出层
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x


# ------------- Video Vision Transformer (简化版 ViT + Time) -------------
class PatchEmbedding(nn.Module):
    """
    把输入帧 (C,H,W) 切成 (N_patches, embed_dim) 向量
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        # 用一个 Conv2d 做 unfold + linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):  # x:[B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2)  # [B, embed_dim, N_patch]
        x = x.transpose(1, 2)  # [B, N_patch, embed_dim]
        return x  # Patch 序列


class VideoTransformer(nn.Module):
    """
    采用时空分离注意力的Video Transformer，灵感来自TimeSformer。
    1. 先在每帧内部做空间注意力。
    2. 再在不同帧之间做时间注意力。
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 n_frames=32,
                 embed_dim=768,
                 depth=8,  # 总深度，会平分给空间和时间
                 n_head=8,
                 mlp_ratio=4.,
                 drop_p=0.1):
        super().__init__()
        self.n_frames = n_frames
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 时空位置编码：需要分别为时间和空间创建
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim)) # 空间位置编码
        self.time_embed = nn.Parameter(torch.zeros(1, n_frames, embed_dim))     # 时间位置编码
        self.pos_drop = nn.Dropout(p=drop_p)

        # 将总深度(depth)平分给空间和时间Transformer
        d_spatial = depth // 2
        d_temporal = depth - d_spatial

        # 空间Transformer编码器
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_head,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_p, batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_encoder_layer, num_layers=d_spatial)

        # 时间Transformer编码器
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_head,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_p, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=d_temporal)

        # 回归头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        # 参数初始化
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):  # x: [B, F, C, H, W]
        B, F, C, H, W = x.shape
        assert F == self.n_frames, "输入帧数必须与模型的 n_frames 一致"

        # 1. Patch Embedding
        # 将 [B, F, C, H, W] -> [B*F, C, H, W]
        x = x.reshape(B * F, C, H, W)
        x = self.patch_embed(x)  # [B*F, N_patches, D]
        n_patches = x.shape[1]

        # 2. 添加CLS Token 和 空间位置编码
        cls_tokens = self.cls_token.expand(B * F, -1, -1)  # [B*F, 1, D]
        x = torch.cat((cls_tokens, x), dim=1) # [B*F, N_patches+1, D]
        x = x + self.pos_embed

        # 3. 空间注意力
        x = self.spatial_transformer(x) # [B*F, N_patches+1, D]

        # 4. 时间注意力准备
        # 将CLS Token分离出来，因为它只参与时间维度的最终预测
        cls_tokens = x[:, 0, :].reshape(B, F, -1) # [B, F, D]

        # 将patch token的形状变回来，准备时间注意力
        x_patches = x[:, 1:, :].reshape(B, F, n_patches, -1) # [B, F, N, D]
        # 维度换位，让时间(F)成为序列长度
        # [B, F, N, D] -> [B, N, F, D] -> [B*N, F, D]
        x_patches = x_patches.permute(0, 2, 1, 3).reshape(B * n_patches, F, -1)

        # 添加时间位置编码
        x_patches = x_patches + self.time_embed

        # 5. 时间注意力
        x_patches = self.temporal_transformer(x_patches) # [B*N, F, D]

        # 6. 数据整合与预测
        # 将处理后的patch在时间维度上取平均，得到一个代表视频动态的特征
        # 另一种方法是也加一个时间的CLS token，这里用平均更简单
        video_feature = x_patches.mean(dim=1).reshape(B, n_patches, -1).mean(dim=1) # [B, D]

        # 将CLS token在时间维度上取平均
        cls_feature = cls_tokens.mean(dim=1) # [B, D]

        # 融合特征（简单相加或拼接）
        # final_feature = (video_feature + cls_feature) / 2
        # final_feature = video_feature
        final_feature = cls_feature

        # 7. 回归头
        pred = self.head(final_feature)
        return pred


class Model(nn.Module):
    """
    Encoder 在主设备 (cuda/mps/cpu)；
    LSTM-decoder 固定在 CPU；其余 decoder 跟随主设备。
    """
    def __init__(self, encoder, decoder, model_type: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model_type = model_type.lower()

    def forward(self, x):
        feats = self.encoder(x)                              # on main device
        dec_device = next(self.decoder.parameters()).device  # cpu for LSTM，其余同 encoder
        feats = feats.to(dec_device)
        out   = self.decoder(feats)
        return out
