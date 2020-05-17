

from os.path import expanduser, join



import sys
print(sys.path)
#sys.path.remove('/usr/local/lib/python3.7/site-packages')
sys.path.append('/usr/local/.pyenv/versions/3.6.0/lib/python3.6/site-packages')
sys.path.append('/Users/kazuya_yufune/.pyenv/versions/3.6.0/lib/python3.6/site-packages')


from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.datasets import PaddedFileSourceDataset, MemoryCacheDataset#これはなに？
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
        files = files[:len(files)-5] # last 5 is real testset

        train_files, test_files = train_test_split(files, test_size=test_size,
                                                   random_state=random_state)
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





for ty in ["acoustic"]:
    for phase in ["train", "test"]:
        train = phase == "train"
        x_dim = duration_linguistic_dim if ty == "duration" else acoustic_linguisic_dim
        y_dim = duration_dim if ty == "duration" else acoustic_dim
        X[ty][phase] = PaddedFileSourceDataset(BinaryFileSource(join(DATA_ROOT, "X_{}".format(ty)),
                                                       dim=x_dim,
                                                       train=train), 
                                               np.max(utt_lengths[ty][phase]))
        Y[ty][phase] = PaddedFileSourceDataset(BinaryFileSource(join(DATA_ROOT, "Y_{}".format(ty)),
                                                       dim=y_dim,
                                                       train=train), 
                                               np.max(utt_lengths[ty][phase]))





print("Total number of utterances:", len(utt_lengths["acoustic"]["train"]))
print("Total number of frames:", np.sum(utt_lengths["acoustic"]["train"]))




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

class PyTorchDataset(torch.utils.data.Dataset):
    """Thin dataset wrapper for pytorch
    
    This does just two things:
        1. On-demand normalization
        2. Returns torch.tensor instead of ndarray
    """
    def __init__(self, X, Y, lengths, X_min, X_max, Y_mean, Y_scale):
        self.X = X
        self.Y = Y
        if isinstance(lengths, list):
            lengths = np.array(lengths)[:,None]
        elif isinstance(lengths, np.ndarray):
            lengths = lengths[:,None]
        self.lengths = lengths
        self.X_min = X_min
        self.X_max = X_max
        self.Y_mean = Y_mean
        self.Y_scale = Y_scale
    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        x = minmax_scale(x, self.X_min, self.X_max, feature_range=(0.01, 0.99))
        y = scale(y, self.Y_mean, self.Y_scale)
        l = torch.from_numpy(self.lengths[idx])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, l
    def __len__(self):
        return len(self.X)


# ##  Model
# 
# We use bidirectional LSTM-based RNNs. Using PyTorch, it's very easy to implement. To handle variable length sequences in mini-batch, we can use [PackedSequence](http://pytorch.org/docs/master/nn.html#torch.nn.utils.rnn.PackedSequence).

# In[183]:


import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tnrange, tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F




class VAE(nn.Module):
    def __init__(self, bidirectional=True, num_layers=1):
        super(VAE, self).__init__()
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1

        self.lstm1 = nn.LSTM(acoustic_linguisic_dim+acoustic_dim, 400, num_layers, bidirectional=bidirectional,  batch_first=True)#入力サイズはここできまる
        self.fc21 = nn.Linear(self.num_direction*400, 1)
        self.fc22 = nn.Linear(self.num_direction*400, 1)
        ##ここまでエンコーダ
        
        self.lstm2 = nn.LSTM(acoustic_linguisic_dim+1, 400, num_layers, bidirectional=bidirectional,  batch_first=True)
        self.fc3 = nn.Linear(self.num_direction*400, acoustic_dim)

    def encode(self, linguistic_f, acoustic_f, mora_index):
        x = torch.cat([linguistic_f, acoustic_f], dim=1)
        out, hc = self.lstm1(x.view(x.size()[0], 1, -1))
        out = out[torch.where(mora_index>0)]
        
        h1 = F.relu(out)

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, linguistic_features, mora_index):
        
        z_tmp = torch.tensor([0]*linguistic_features.size()[0], dtype=torch.float32)
        count = 0
        for mora_i in mora_index.numpy():
            if mora_i == 1:
                z_tmp[int(mora_i)] = z[count]
                count += 1

        
        x = torch.cat([linguistic_features, z_tmp.view(-1, 1)], dim=1).view(linguistic_features.size()[0], 1, -1)
        
        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3)
        
        return torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features, acoustic_features, mora_index):
        mu, logvar = self.encode(linguistic_features, acoustic_features, mora_index)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z, linguistic_features, mora_index), mu, logvar


model = VAE().to('cpu')



# In[104]:


import pandas as pd


# In[259]:


mora_index_lists = sorted(glob(join('data/NIT-ATR503/mora_index', "*.csv")))[:100]
mora_index_lists = mora_index_lists[:len(mora_index_lists)-5] # last 5 is real testset

mora_index_lists_for_model = [np.array(pd.read_csv(path)).reshape(-1) for path in mora_index_lists]



train_mora_index_lists, test_mora_index_lists = train_test_split(mora_index_lists_for_model, test_size=test_size,
                                                  random_state=random_state)







for i in range(90):
    print(np.array(pd.read_csv(mora_index_lists[i])).reshape(-1).shape[0] / X['acoustic']['train'][i].shape[0])





device='cuda'
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1), x.view(-1, ), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


func_tensor = np.vectorize(torch.from_numpy)

X_acoustic_train = [torch.from_numpy(X['acoustic']['train'][i]) for i in range(len(X['acoustic']['train']))] 
Y_acoustic_train = [torch.from_numpy(Y['acoustic']['train'][i]) for i in range(len(Y['acoustic']['train']))]
train_mora_index_lists = [torch.tensor(train_mora_index_lists[i]) for i in range(len(train_mora_index_lists))]

X_acoustic_test = [torch.from_numpy(X['acoustic']['test'][i]) for i in range(len(X['acoustic']['test']))]
Y_acoustic_test = [torch.from_numpy(Y['acoustic']['test'][i]) for i in range(len(Y['acoustic']['test']))]
test_mora_index_lists = [torch.tensor(test_mora_index_lists[i]) for i in range(len(test_mora_index_lists))]

train_loader = [[X_acoustic_train[i], Y_acoustic_train[i], train_mora_index_lists[i]] for i in range(len(train_mora_index_lists))]
test_loader = [[X_acoustic_test[i], Y_acoustic_test[i], test_mora_index_lists[i]] for i in range(len(test_mora_index_lists))]


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        for j in range(3):
            data[j] = data[j].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data[0], data[1], data[2])
        loss = loss_function(recon_batch, data[1], mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / 1))
    
    return train_loss


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data, in enumerate(test_loader):
            for j in range(3):
                data[j] = data[j].to(device)
            recon_batch, mu, logvar = model(data[0], data[1], data[2])
            test_loss += loss_function(recon_batch, data[1], mu, logvar).item()
            """
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            """

    test_loss /= 1
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    return test_loss






loss_list = []
test_loss_list = []
num_epochs = 5

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

# save the training model
np.save('loss_list.npy', np.array(loss_list))
np.save('test_loss_list.npy', np.array(test_loss_list))
torch.save(model.state_dict(), 'vae.pth')


# ## Train
# 
# ### Configurations
# 
# Network hyper parameters and training configurations (learning rate, weight decay, etc).

# In[200]:
