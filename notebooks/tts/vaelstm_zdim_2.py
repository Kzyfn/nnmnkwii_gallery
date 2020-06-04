

from os.path import expanduser, join



import sys
print(sys.path)
#sys.path.remove('/usr/local/lib/python3.7/site-packages')
sys.path.append('/usr/local/.pyenv/versions/3.6.0/lib/python3.6/site-packages')
sys.path.append('/Users/kazuya_yufune/.pyenv/versions/3.6.0/lib/python3.6/site-packages')

import time

from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.datasets import PaddedFileSourceDataset, MemoryCacheDataset#これはなに
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii import paramgen
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter

from os.path import join, expanduser, basename, splitext, basename, exists
import os
from glob import glob
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
import librosa
import librosa.display




DATA_ROOT = "./data/basic5000"#NIT-ATR503/"#
test_size = 0.01 # This means 480 utterances for training data
random_state = 1234



mgc_dim = 180#メルケプストラム次数　？？
lf0_dim = 3#対数fo　？？ なんで次元が３？
vuv_dim = 1#無声or 有声フラグ　？？
bap_dim = 15#発話ごと非周期成分　？？

duration_linguistic_dim = 438#question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 442#上のやつ+frame_features とは？？
duration_dim = 1
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim #aoustice modelで求めたいもの

fs = 48000
frame_period = 5
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
hop_length = int(0.001 * frame_period * fs)

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

use_phone_alignment = True
acoustic_subphone_features = "coarse_coding" if use_phone_alignment else "full" #とは？



class BinaryFileSource(FileDataSource):
    def __init__(self, data_root, dim, train):
        self.data_root = data_root
        self.dim = dim
        self.train = train
    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*.bin")))
        #files = files[:len(files)-5] # last 5 is real testset
        train_files = []
        test_files = []
        #train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

        for i, path in enumerate(files):
            if (i - 1) % 20 == 0:#test
                pass
            elif i % 20 == 0:#valid
                test_files.append(path)
            else:
                train_files.append(path)

        if self.train:
            return train_files
        else:
            return test_files
    def collect_features(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, self.dim)


X = {"acoustic": {}}
Y = {"acoustic": {}}
utt_lengths = { "acoustic": {}}
for ty in ["acoustic"]:
    for phase in ["train", "test"]:
        train = phase == "train"
        x_dim = duration_linguistic_dim if ty == "duration" else acoustic_linguisic_dim
        y_dim = duration_dim if ty == "duration" else acoustic_dim
        X[ty][phase] = FileSourceDataset(BinaryFileSource(join(DATA_ROOT, "X_{}".format(ty)),
                                                       dim=x_dim,
                                                       train=train))
        Y[ty][phase] = FileSourceDataset(BinaryFileSource(join(DATA_ROOT, "Y_{}".format(ty)),
                                                       dim=y_dim,
                                                       train=train))
        utt_lengths[ty][phase] = np.array([len(x) for x in X[ty][phase]], dtype=np.int)





X_min = {}
X_max = {}
Y_mean = {}
Y_var = {}
Y_scale = {}

for typ in ["acoustic"]:
    X_min[typ], X_max[typ] = minmax(X[typ]["train"], utt_lengths[typ]["train"])
    Y_mean[typ], Y_var[typ] = meanvar(Y[typ]["train"], utt_lengths[typ]["train"])
    Y_scale[typ] = np.sqrt(Y_var[typ])




from torch.utils import data as data_utils


import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tnrange, tqdm
from torch import optim
import torch.nn.functional as F


z_dim = 4
dropout = 0.3
num_layers = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAE(nn.Module):
    def __init__(self, bidirectional=True, num_layers=num_layers):
        super(VAE, self).__init__()
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1

        self.lstm1 = nn.LSTM(acoustic_linguisic_dim+acoustic_dim, 400, num_layers, bidirectional=bidirectional, dropout=dropout)#入力サイズはここできまる
        self.fc21 = nn.Linear(self.num_direction*400, z_dim)
        self.fc22 = nn.Linear(self.num_direction*400, z_dim)
        ##ここまでエンコーダ
        
        self.lstm2 = nn.LSTM(acoustic_linguisic_dim+z_dim, 400, num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc3 = nn.Linear(self.num_direction*400, acoustic_dim)

    def encode(self, linguistic_f, acoustic_f, mora_index):
        x = torch.cat([linguistic_f, acoustic_f], dim=1)
        out, hc = self.lstm1(x.view( x.size()[0],1, -1))
        nonzero_indices = torch.nonzero(mora_index.view(-1).data).squeeze()
        out = out[nonzero_indices]
        del nonzero_indices
        
        h1 = F.relu(out)

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, linguistic_features, mora_index):
        
        z_tmp = torch.tensor([[0]*z_dim]*linguistic_features.size()[0], dtype=torch.float32, requires_grad=True).to(device)
        count = 0
        prev_index = 0
        for i, mora_i in enumerate(mora_index):
            if mora_i == 1:
                z_tmp[prev_index:i] = z[count]
                prev_index = i
                count += 1
                
                

        
        x = torch.cat([linguistic_features, z_tmp.view(-1, z_dim)], dim=1).view(linguistic_features.size()[0], 1, -1)
        
        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3)
        
        return self.fc3(h3)#torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features, acoustic_features, mora_index):
        mu, logvar = self.encode(linguistic_features, acoustic_features, mora_index)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z, linguistic_features, mora_index), mu, logvar






# In[104]:


import pandas as pd


mora_index_lists = sorted(glob(join('data/basic5000/mora_index', "*.csv")))
#mora_index_lists = mora_index_lists[:len(mora_index_lists)-5] # last 5 is real testset
mora_index_lists_for_model = [np.array(pd.read_csv(path)).reshape(-1) for path in mora_index_lists]

train_mora_index_lists = []
test_mora_index_lists = []
#train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

for i, mora_i in enumerate(mora_index_lists_for_model):
    if (i - 1) % 20 == 0:#test
        pass
    elif i % 20 == 0:#valid
        test_mora_index_lists.append(mora_i)
    else:
        train_mora_index_lists.append(mora_i)



model = VAE().to(device)
model.load_state_dict(torch.load('1layers_zdim4/vae_mse30.pth'))
optimizer = optim.Adam(model.parameters(), lr=2e-3)#1e-3

start = time.time()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x.view(-1), x.view(-1, ), reduction='sum')#F.binary_cross_entropy(recon_x.view(-1), x.view(-1, ), reduction='sum')
    #print('LOSS')
    #print(BCE)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print(KLD)
    return MSE +  KLD


func_tensor = np.vectorize(torch.from_numpy)

X_acoustic_train = [X['acoustic']['train'][i] for i in range(len(X['acoustic']['train']))] 
Y_acoustic_train = [Y['acoustic']['train'][i] for i in range(len(Y['acoustic']['train']))]
train_mora_index_lists = [train_mora_index_lists[i] for i in range(len(train_mora_index_lists))]

train_num = len(X_acoustic_train)

X_acoustic_test = [X['acoustic']['test'][i] for i in range(len(X['acoustic']['test']))]
Y_acoustic_test = [Y['acoustic']['test'][i] for i in range(len(Y['acoustic']['test']))]
test_mora_index_lists = [test_mora_index_lists[i] for i in range(len(test_mora_index_lists))]

train_loader = [[X_acoustic_train[i], Y_acoustic_train[i], train_mora_index_lists[i]] for i in range(len(train_mora_index_lists))]
test_loader = [[X_acoustic_test[i], Y_acoustic_test[i], test_mora_index_lists[i]] for i in range(len(test_mora_index_lists))]


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        tmp = []

        
        for j in range(3):
            tmp.append(torch.from_numpy(data[j]).to(device))


        optimizer.zero_grad()
        recon_batch, mu, logvar = model(tmp[0], tmp[1], tmp[2])
        loss = loss_function(recon_batch, tmp[1], mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        del tmp
        if batch_idx % 4945 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, train_num,
                100. * batch_idx / train_num,
                loss.item()))


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader)))
    
    return train_loss / len(train_loader)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data, in enumerate(test_loader):
            tmp = []

     
            for j in range(3):
                tmp.append(torch.tensor(data[j]).to(device))


            recon_batch, mu, logvar = model(tmp[0], tmp[1], tmp[2])
            test_loss += loss_function(recon_batch, tmp[1], mu, logvar).item()

            del tmp

    test_loss /= len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    return test_loss






loss_list = []
test_loss_list = []
num_epochs = 30

#model.load_state_dict(torch.load('vae.pth'))

for epoch in range(1, num_epochs + 1):
    loss = train(epoch)
    test_loss = test(epoch)
    print(loss)
    print(test_loss)

    print('epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}'.format(
        epoch + 1,
        num_epochs,
        loss,
        test_loss))

    # logging
    loss_list.append(loss)
    test_loss_list.append(test_loss)

    print(time.time() - start)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), str(num_layers) +'layers_zdim' + str(z_dim) +'/vae_mse'+str(epoch+30)+'.pth')
    np.save(str(num_layers) +'layers_zdim' + str(z_dim) +'/loss_listz_2dim_.npy', np.array(loss_list))
    np.save(str(num_layers) +'layers_zdim' + str(z_dim) +'/test_loss_listz_2dim_.npy', np.array(test_loss_list))

# save the training model
np.save(str(num_layers) +'layers_zdim' + str(z_dim) +'/loss_list_lstm1layer_z2dim.npy', np.array(loss_list))
np.save(str(num_layers) +'layers_zdim' + str(z_dim) +'/test_loss_list_lstm1layer_z2dim.npy', np.array(test_loss_list))
torch.save(model.state_dict(), str(num_layers) +'layers_zdim' + str(z_dim) +'/vae_mse_0.vae_mse_z_2dim.pth')
