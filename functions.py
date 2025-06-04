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
        skip_frame = frame_count // total_frames
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
    def __init__(self, CNN_embed_dim=256, h_RNN_layers=1, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=5):
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

class Model(nn.Module):
    def __init__(self, EncoderCNN, DecoderRNN):
        super().__init__()
        self.EncoderCNN = EncoderCNN
        self.DecoderRNN = DecoderRNN

    def forward(self, x):
        return self.DecoderRNN(self.EncoderCNN(x).to('cpu')).to('mps')
