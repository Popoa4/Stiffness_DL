import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import Model, EncoderCNN, DecoderRNN, Dataset_CRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle


# set path
data_path = "/Users/ethanshao/Desktop/ucl/research project/Stiffness_DL/Dataset/hardness-test"    # define UCF-101 RGB data path
#data_path = "E:/deeplearning/Project/Datasets/hardness-2016/hardness-test"
hardness_name_path = './hardness.pkl'
save_model_path = "./CRNN_ckpt5/"

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 512, 348  #1024,768   #1024,768  512, 348
CNN_embed_dim = 256 #512  #256    # latent dim extracted by 2D CNN
img_x, img_y = 224, 224  # resize video 2d frame size
dropout_p = 0.1          # dropout probability 0.5

# DecoderRNN architecture
RNN_hidden_layers = 1
RNN_hidden_nodes = 256 #512    #256
RNN_FC_dim = 128# 256         #128

# training parameters

k = 5         #101    # number of target category
epochs = 100      #100   # training epochs
batch_size = 16    #16
learning_rate = 1e-4  # 1e-4
log_interval = 10   # interval for displaying training info 显示训练时间间隔

# Select which frame to begin & end in videos
# begin_frame, end_frame, skip_frame = 1, 50, 5

def train(model, device, train_loader, optimizer, epoch, mean_sigma):
    # set model as training mode
    model.train()
    mean, sigma = mean_sigma[0], mean_sigma[1]
    MSE_Loss = torch.nn.MSELoss()
    losses = []
    N_count = 0   # counting total trained sample in one epoch
    error = []
    for batch_idx, (X, y_scale, y) in enumerate(train_loader):
        # distribute data to device
        X, y_scale, y = X.to(device), y_scale.to(device), y.numpy()

        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)   # output has dim = (batch, number of classes)

        loss = MSE_Loss(output, y_scale)
        loss = torch.sqrt(loss)
        # for name, parms in model.named_parameters():
        #     if name == "EncoderCNN.conv1.0.weight":
        #         print('-->name:', name)
        #         print('-->para:', parms)
        #         print('-->grad_requirs:', parms.requires_grad)
        #         print('-->grad_value:', parms.grad)
        #         print("===")

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # print("=============更新之后===========")
        #
        # for name, parms in model.named_parameters():
        #     if name == "EncoderCNN.conv1.0.weight":
        #         print('-->name:', name)
        #         print('-->para:', parms)
        #         print('-->grad_requirs:', parms.requires_grad)
        #         print('-->grad_value:', parms.grad)
        #         print("===")

        # show information
        # if (batch_idx + 1) % log_interval == 0:
        pred = (output.detach().cpu().numpy() * sigma + mean)
        distance = np.mean((np.abs(pred - y) / y).squeeze())
        error.append(distance)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\trmse_Loss: {:.6f}, avg_error: {:.2f}'.format(
            epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), 100. * distance))
    average_error = np.mean(error)
    average_loss = np.mean(losses)
    return average_loss, average_error


def validation(model, device, optimizer, test_loader, mean_sigma):
    # set model as testing mode
    model.eval()
    mean, sigma = mean_sigma[0], mean_sigma[1]
    MSE_Loss = torch.nn.MSELoss()
    test_loss = 0
    error = []
    pred_list = []
    true_label = []
    i = 0
    with torch.no_grad():
        for X, y_scale, y in test_loader:
            # distribute data to device
            X, y_scale, y = X.to(device), y_scale.to(device), y.numpy()
            i = i + 1
            output = model(X)
            loss = MSE_Loss(output, y_scale)
            loss = torch.sqrt(loss)
            pred = (output.detach().cpu().numpy() * sigma + mean)
            pred_list.append(pred)
            true_label.append(y)
            distance = np.abs(pred - y) / y
            error.append(np.array([distance.squeeze()]))
            test_loss += loss.item()                 # sum up batch loss
    error = np.concatenate(error, axis=0)
    average_error = np.sum(error) / len(test_loader.dataset)
    test_loss = test_loss / i
    pred_list = np.concatenate(pred_list, axis=0).squeeze()
    true_label = np.concatenate(true_label, axis=0).squeeze()
    # show information
    print('\nTest set ({:d} samples): rmse_loss: {:.4f}, avg_error: {:.2f}\n'.format(len(test_loader.dataset), test_loss, 100 * average_error))

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, 'model{}.pth'.format(epoch + 1)))  # save spatial_encoder
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, average_error, pred_list, true_label


# Detect devices
use_cuda = torch.cuda.is_available() or torch.backends.mps.is_available()            # check if GPU exists
# device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU or MPS (Apple Silicon)
# use CPU or GPU or MPS (Apple Silicon)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Data loading parameters
#params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': False} if use_cuda else {}
# load  hardness names
# with open(hardness_name_path, 'rb') as f:
#     hardness_label = pickle.load(f)

all_Y_list = []
fnames = os.listdir(data_path)

all_X_list = []
for f in fnames:
    label = int(f[8: 10])
    all_Y_list.append(label)
    all_X_list.append(f)



# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_Y_list, test_size=0.2, random_state=42)

transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# transform = transforms.Compose([transforms.Resize([img_x, img_y]),
#                                  transforms.ToTensor()])
# transform = transforms.Compose([transforms.CenterCrop((img_x, img_y)),
#                                   transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# transform = transforms.Compose([transforms.CenterCrop((img_x, img_y)),
#                                   transforms.ToTensor()])

train_label = np.array(train_label, dtype=np.float32).reshape(-1, 1)
test_label = np.array(test_label, dtype=np.float32).reshape(-1, 1)

# 初始化一个 StandardScaler 对象
scaler_train = StandardScaler()
scaler_test = StandardScaler()
# 使用 fit_transform 方法对 Y 进行拟合和转换
train_label_scaled = scaler_train.fit_transform(train_label)
test_label_scaled = scaler_test.fit_transform(test_label)
train_mean = (scaler_train.mean_).astype(np.float32)
train_sigma = scaler_train.scale_.astype(np.float32)
test_mean = scaler_test.mean_.astype(np.float32)
test_sigma = scaler_test.scale_.astype(np.float32)

# np.save('test_mean.npy', test_mean)

#selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label_scaled, train_label, transform=transform), \
                       Dataset_CRNN(data_path, test_list, test_label_scaled, test_label, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

# Create model
cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim)

rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k)

model = Model(cnn_encoder, rnn_decoder).to(device)
#model.load_state_dict(torch.load("./CRNN_ckpt5/model4.10.pth"))

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# record training process
epoch_train_losses = []
epoch_train_error = []
epoch_test_losses = []
epoch_test_error = []
epoch_pred_list = []
epoch_true_label = []

# start training
for epoch in range(epochs):
    # train, test model
    train_loss, train_error = train(model, device, train_loader, optimizer, epoch, mean_sigma=[train_mean, train_sigma])
    test_loss, test_error, pred_list, true_label = validation(model, device, optimizer, valid_loader, mean_sigma=[test_mean, test_sigma])

    # # save results
    epoch_train_losses.append(train_loss)
    epoch_train_error.append(train_error)
    epoch_test_losses.append(test_loss)
    epoch_test_error.append(test_error)
    epoch_pred_list.append(pred_list)
    epoch_true_label.append(true_label)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_error)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_error)
    E = np.array(epoch_pred_list)
    F = np.array(epoch_true_label)

    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_error.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_error.npy', D)
    np.save('./CRNN_epoch_pred_list.npy', E)
    np.save('./CRNN_epoch_true_label.npy', F)

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A)  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('rmse_loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B)  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
plt.title("regression error")
plt.xlabel('epochs')
plt.ylabel('error')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_hardness5_CRNN.png"
plt.savefig(title, dpi=600)
plt.show()


