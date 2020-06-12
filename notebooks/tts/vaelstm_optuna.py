

from os.path import expanduser, join
import os
import argparse
import optuna
def parse():
    parser = argparse.ArgumentParser(description='LSTM VAE', )
    parser.add_argument(
        '-ne',
        '--num_epoch',
        type=int,
        default=30,
    )
    parser.add_argument(
        '-od',
        '--output_dir',
        type=str,
        required=True,
    ),
    parser.add_argument(
        '-dr',
        '--dropout_ratio',
        type=float,
        default=0.3,
    ),
    parser.add_argument(
        '-mp',
        '--model_path',
        type=str,
        default='',
    ),
    parser.add_argument(
        '-tr',
        '--train_ratio',
        type=float,
        default=1.0
    ),
    parser.add_argument(
        '-nt',
        '--num_trials',
        type=int,
        required=True
    )

    return parser.parse_args()

args = parse()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)



import sys

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


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))
 
def calc_lf0_rmse(natural, generated, lf0_idx, vuv_idx):
    idx = (natural[:, vuv_idx] * (generated[:, vuv_idx] >= 0.5)).astype(bool)
    return rmse(natural[idx, lf0_idx], generated[idx, lf0_idx]) * 1200 / np.log(2)  # unit: [cent]



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



dropout= args.dropout_ratio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAE(nn.Module):
    def __init__(self, num_layers, z_dim, bidirectional=True):
        super(VAE, self).__init__()
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1
        self.z_dim = z_dim
        self.fc11 = nn.Linear(acoustic_linguisic_dim+acoustic_dim, acoustic_linguisic_dim+acoustic_dim)

        self.lstm1 = nn.LSTM(acoustic_linguisic_dim+acoustic_dim, 400, num_layers, bidirectional=bidirectional, dropout=dropout)#入力サイズはここできまる
        self.fc21 = nn.Linear(self.num_direction*400, z_dim)
        self.fc22 = nn.Linear(self.num_direction*400, z_dim)
        ##ここまでエンコーダ
        
        self.fc12 = nn.Linear(acoustic_linguisic_dim+z_dim, acoustic_linguisic_dim+z_dim)
        self.lstm2 = nn.LSTM(acoustic_linguisic_dim+z_dim, 400, 2, bidirectional=bidirectional, dropout=dropout)
        self.fc3 = nn.Linear(self.num_direction*400, acoustic_dim)

    def encode(self, linguistic_f, acoustic_f, mora_index):
        x = torch.cat([linguistic_f, acoustic_f], dim=1)
        x = self.fc11(x)
        x = F.relu(x)

        out, hc = self.lstm1(x.view( x.size()[0],1, -1))
        out = out[mora_index]
        
        h1 = F.relu(out)

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, linguistic_features, mora_index):
        
        z_tmp = torch.tensor([[0]*self.z_dim]*linguistic_features.size()[0], dtype=torch.float32, requires_grad=True).to(device)
        
        for i, mora_i in enumerate(mora_index):
            prev_index = 0 if i == 0 else mora_index[i-1]
            z_tmp[prev_index:mora_i] = z[i]
     

        
        x = torch.cat([linguistic_features, z_tmp.view(-1, self.z_dim)], dim=1)
        x = self.fc12(x)
        x = F.relu(x)

        h3, (h, c) = self.lstm2(x.view(linguistic_features.size()[0], 1, -1))
        h3 = F.relu(h3)
        
        return self.fc3(h3)#torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features, acoustic_features, mora_index):
        mu, logvar = self.encode(linguistic_features, acoustic_features, mora_index)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z, linguistic_features, mora_index), mu, logvar





#model.load_state_dict(torch.load('vae_mse_0.01kld_z_changed_losssum_batchfirst_10.pth'))
# In[104]:


import pandas as pd

def objective(trial):
    mora_index_lists = sorted(glob(join('data/basic5000/mora_index/squeezed_', "*.csv")))
    #mora_index_lists = mora_index_lists[:len(mora_index_lists)-5] # last 5 is real testset
    mora_index_lists_for_model = [np.loadtxt(path).reshape(-1) for path in mora_index_lists]
    print(mora_index_lists[0])
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


    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 4)
    z_dim = trial.suggest_categorical('z_dim', [1, 2, 4, 8, 16, 32])

    model = VAE(num_lstm_layers, z_dim).to(device)

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


    train_ratio = int(args.train_ratio*len(train_mora_index_lists))#1

    X_acoustic_train = [X['acoustic']['train'][i] for i in range(len(X['acoustic']['train']))][:train_ratio]
    Y_acoustic_train = [Y['acoustic']['train'][i] for i in range(len(Y['acoustic']['train']))][:train_ratio]
    train_mora_index_lists = [train_mora_index_lists[i] for i in range(len(train_mora_index_lists))][:train_ratio]

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

            #del tmp

            if batch_idx % len(train_loader) == 0:
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
        f0_loss = 0
        with torch.no_grad():
            for i, data, in enumerate(test_loader):
                tmp = []

        
                for j in range(3):
                    tmp.append(torch.tensor(data[j]).to(device))


                recon_batch, z, z_unquantized = model(tmp[0], tmp[1], tmp[2])
                test_loss += loss_function(recon_batch, tmp[1],  z, z_unquantized).item()
                f0_loss += calc_lf0_rmse(recon_batch.cpu().numpy().reshape(-1, 199), tmp[1].cpu().numpy().reshape(-1, 199), lf0_start_idx, vuv_start_idx)
                #del tmp

        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        
        return test_loss, f0_loss






    loss_list = []
    test_loss_list = []
    test_f0_erros = []

    num_epochs = args.num_epoch


    for epoch in range(1, num_epochs + 1):
        loss = train(epoch)
        test_loss, f0_loss = test(epoch)

        if epoch == 1:
            min_f0_loss = f0_loss
            min_f0_epoch = epoch
        elif f0_loss < min_f0_loss:
            min_f0_loss = f0_loss
            min_f0_epoch = epoch

        print('epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}'.format(
            epoch + 1,
            num_epochs,
            loss,
            test_loss))

        #logging
        loss_list.append(loss)
        test_loss_list.append(test_loss)
        test_f0_erros.append(f0_loss)

        print(time.time() - start)


        if epoch % 5 == 0:
            torch.save(model.state_dict(),  '{}/{}layers_zdim{}_model_{}.pth'.format(args.output_dir, num_lstm_layers, z_dim, epoch) )
        np.save(args.output_dir +'/{}layers_zdim{}_loss_list.npy'.format(num_lstm_layers, z_dim), np.array(loss_list))
        np.save(args.output_dir +'/{}layers_zdim{}_test_loss_list.npy'.format(num_lstm_layers, z_dim), np.array(test_loss_list))
        np.save(args.output_dir +'/{}layers_zdim{}_test_f0_loss_list.npy'.format(num_lstm_layers, z_dim), np.array(test_f0_erros))

        if epoch > min_f0_epoch + 4:
            torch.save(model.state_dict(),  '{}/{}layers_zdim{}_model_{}.pth'.format(args.output_dir, num_lstm_layers, z_dim, epoch))
            break

    return f0_loss


study = optuna.create_study()
study.optimize(objective, n_trials = args.num_trials)