import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# set path
data_path = "./jpegs_256/"    # define UCF-101 RGB data path
action_name_path = "./UCF101actions.pkl"
save_model_path = "./CRNN_ckpt/"

# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
img_x, img_y = 256, 342  # resize video 2d frame size
dropout_p = 0.0       # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category
batch_size = 40
begin_frame, end_frame, skip_frame = 1, 29, 1

with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)

le = LabelEncoder()
le.fit(action_names)
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

actions = []
fnames = os.listdir(data_path)
all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])
    all_names.append(f)

all_X_list = all_names
all_y_list = labels2cat(le, actions)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if torch.backends.mps.is_available() or torch.cuda.is_available() else {}

transform = transforms.Compose([
    transforms.Resize([img_x, img_y]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
all_data_loader = data.DataLoader(
    Dataset_CRNN(data_path, all_X_list, all_y_list, selected_frames, transform=transform),
    **params
)

cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers,
                         h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch41.pth')))
rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch41.pth')))
print('CRNN model reloaded!')

print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
all_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, all_data_loader)

df = pd.DataFrame(data={
    'filename': fnames,
    'y': cat2labels(le, all_y_list),
    'y_pred': cat2labels(le, all_y_pred)
})
df.to_pickle("./UCF101_videos_prediction.pkl")
print('video prediction finished!')
