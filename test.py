import torch
from functions import Model, EncoderCNN, DecoderRNN, Dataset_CRNN
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# device = "cuda:0"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
data_path = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness_5_2024_03_03"
hardness_name_path = './hardness.pkl'
save_model_path = "./CRNN_ckpt1/"

CNN_fc_hidden1, CNN_fc_hidden2 = 512, 348
CNN_embed_dim = 256
img_x, img_y = 256, 342
dropout_p = 0.0

RNN_hidden_layers = 1
RNN_hidden_nodes = 256
RNN_FC_dim = 128

k = 5
epochs = 2
batch_size = 16
learning_rate = 1e-4
log_interval = 10

begin_frame, end_frame, skip_frame = 1, 50, 5
all_Y_list = []
fnames = os.listdir(data_path)

all_X_list = []
for f in fnames:
    label = int(f[7: 9])
    all_Y_list.append(label)
    all_X_list.append(f)

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': False}
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_Y_list, test_size=0.2, random_state=42)

transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_label = np.array(train_label, dtype=np.float32).reshape(-1, 1)
test_label = np.array(test_label, dtype=np.float32).reshape(-1, 1)

scaler_train = StandardScaler()
scaler_test = StandardScaler()
train_label_scaled = scaler_train.fit_transform(train_label)
test_label_scaled = scaler_test.fit_transform(test_label)

train_mean = (scaler_train.mean_).astype(np.float32)
train_sigma = scaler_train.scale_.astype(np.float32)
test_mean = scaler_test.mean_.astype(np.float32)
test_sigma = scaler_test.scale_.astype(np.float32)

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label_scaled, train_label, selected_frames, transform=transform),                        Dataset_CRNN(data_path, test_list, test_label_scaled, test_label, selected_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k)
model = Model(cnn_encoder, rnn_decoder).to()
