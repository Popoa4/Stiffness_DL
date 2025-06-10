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
    def __init__(self, data_path, folders, labels, gt, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.gt_labels = gt
        self.folders = folders
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        folder_path = os.path.join(path, selected_folder)
        frame_files = sorted(os.listdir(folder_path))
        frame_count = len(frame_files)
        begin_frame = 0
        end_frame = frame_count - 1
        total_frames = 10
        if frame_count < total_frames:
            begin_frame = 0
            end_frame = frame_count - 1
            total_frames = frame_count
        skip_frame = frame_count // total_frames
        # print("Reading folder: {}, total frames: {}, skip frame: {}".format(
        #     selected_folder, frame_count, skip_frame))
        selected_frames = np.arange(begin_frame, end_frame + 1, skip_frame)
        if len(selected_frames) > total_frames:
            selected_frames = selected_frames[:total_frames]
        selected_frames = selected_frames.tolist()
        for i in selected_frames:
            img_path = os.path.join(folder_path, 'frame_{:04d}.jpg'.format(i))
            image = Image.open(img_path)
            if use_transform is not None:
                image = use_transform(image)
            X.append(image)
        return torch.stack(X, dim=0)

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


# class Model(nn.Module):
#     def __init__(self, EncoderCNN, Decoder):
#         super().__init__()
#         self.EncoderCNN = EncoderCNN
#         self.Decoder = Decoder
#
#     def forward(self, x):
#         # LSTM: Ensure the input is on the CPU for compatibility with MPS
#         # return self.Decoder(self.EncoderCNN(x).to('cpu')).to('mps')
#         # GRU: all on mps
#         # transformer: all on mps
#         return self.Decoder(self.EncoderCNN(x))
# 其余内容完全不变，只贴出 Model

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
