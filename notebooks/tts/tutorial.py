
from os.path import expanduser, join


import matplotlib.pyplot as plt

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




DATA_ROOT = "./data/basic5000"#NIT-ATR503/"
test_size = 0.01#0.01 # This means 480 utterances for training data
random_state = 1234




mgc_dim = 180#メルケプストラム次数　？？
lf0_dim = 3#対数fo　？？ なんで次元が３？
vuv_dim = 1#無声or 有声フラグ　？？
bap_dim = 15#発話ごと非周期成分　？？

duration_linguistic_dim = 531#question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 535#上のやつ+frame_features とは？？
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
        files = files[:len(files)-5]#[:10] # last 5 is real testset
        train_files, test_files = train_test_split(files, test_size=test_size,
                                                   random_state=random_state)
        if self.train:
            return train_files
        else:
            return test_files
    def collect_features(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, self.dim)



X = {"duration":{}, "acoustic": {}}
Y = {"duration":{}, "acoustic": {}}
utt_lengths = {"duration":{}, "acoustic": {}}
for ty in ["duration", "acoustic"]:
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



for ty in ["duration", "acoustic"]:
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









X_min = {}
X_max = {}
Y_mean = {}
Y_var = {}
Y_scale = {}

for typ in ["acoustic", "duration"]:
    X_min[typ], X_max[typ] = minmax(X[typ]["train"], utt_lengths[typ]["train"])
    Y_mean[typ], Y_var[typ] = meanvar(Y[typ]["train"], utt_lengths[typ]["train"])
    Y_scale[typ] = np.sqrt(Y_var[typ])





# ### Combine datasets and normalization.
# 
# In this demo we use PyTorch to build DNNs. `PyTorchDataset` is just a glue Dataset wrapper, which combines our dataset and normalization. Note that since our dataset is zero-padded, we need its lengths.

# In[19]:


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


import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tnrange, tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim


# In[21]:


class MyRNN(nn.Module):
    def __init__(self, D_in, H, D_out, num_layers=1, bidirectional=True):
        super(MyRNN, self).__init__()
        self.hidden_dim = H
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1
        self.lstm = nn.LSTM(D_in, H, num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden2out = nn.Linear(self.num_direction*self.hidden_dim, D_out)
        
    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)), 
                Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)))
        return h,c

    def forward(self, sequence, lengths, h, c):
        sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)
        return output



num_hidden_layers = 3
hidden_size = 512

batch_size = 4

n_workers = 2
pin_memory = True
nepoch = 10
lr = 0.002
weight_decay = 1e-6
use_cuda = torch.cuda.is_available()





def train_rnn(model, optimizer, X, Y, X_min, X_max, Y_mean, Y_scale,
          utt_lengths, cache_size=1000):
    if use_cuda:
        model = model.cuda()
        
    X_train, X_test = X["train"], X["test"]
    Y_train, Y_test = Y["train"], Y["test"]
    X_train[:, 286:336, :] = 0
    X_test[:, 286:336, :] = 0
    train_lengths, test_lengths = utt_lengths["train"], utt_lengths["test"]
    
    # Sequence-wise train loader
    X_train_cache_dataset = MemoryCacheDataset(X_train, cache_size)
    Y_train_cache_dataset = MemoryCacheDataset(Y_train, cache_size)
    train_dataset = PyTorchDataset(X_train_cache_dataset, Y_train_cache_dataset, train_lengths,
                                  X_min, X_max, Y_mean, Y_scale)
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    
    # Sequence-wise test loader
    X_test_cache_dataset = MemoryCacheDataset(X_test, cache_size)
    Y_test_cache_dataset = MemoryCacheDataset(Y_test, cache_size)
    test_dataset = PyTorchDataset(X_test_cache_dataset, Y_test_cache_dataset, test_lengths,
                                 X_min, X_max, Y_mean, Y_scale)
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    
    dataset_loaders = {"train": train_loader, "test": test_loader}
        
    # Training loop
    criterion = nn.MSELoss()
    model.train()
    print("Start utterance-wise training...")
    loss_history = {"train": [], "test": []}
    for epoch in tnrange(nepoch):
        for phase in ["train", "test"]:
            running_loss = 0
            for x, y, lengths in dataset_loaders[phase]:
                # Sort by lengths . This is needed for pytorch's PackedSequence
                sorted_lengths, indices = torch.sort(lengths.view(-1), dim=0, descending=True)
                sorted_lengths = sorted_lengths.long().numpy()
                # Get sorted batch
                x, y = x[indices], y[indices]
                # Trim outputs with max length
                y = y[:, :sorted_lengths[0]]
                
                # Init states
                h, c = model.init_hidden(len(sorted_lengths))
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                if use_cuda:
                    h, c = h.cuda(), c.cuda()
                x, y = Variable(x), Variable(y)
                optimizer.zero_grad()
                
                # Do apply model for a whole sequence at once
                # no need to keep states
                y_hat = model(x, sorted_lengths, h, c)
                loss = criterion(y_hat, y)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            loss_history[phase].append(running_loss / (len(dataset_loaders[phase])))
        
    return loss_history




models = {}
for typ in ["duration", "acoustic"]:
    models[typ] = MyRNN(X[typ]["train"][0].shape[-1],
                            hidden_size, Y[typ]["train"][0].shape[-1],
                            num_hidden_layers, bidirectional=True)
    print("Model for {}\n".format(typ), models[typ])



#models['duration'].load_state_dict(torch.load('speechsynthesis_models/duration_state_dict', map_location=torch.device('cpu')))
#models['acoustic'].load_state_dict(torch.load("speechsynthesis_models/acoustic_state_dict", map_location=torch.device('cpu')))


# ### Training Duration model

# In[39]:


ty = "duration"
optimizer = optim.Adam(models[ty].parameters(), lr=lr, weight_decay=weight_decay)
loss_history_1 = train_rnn(models[ty], optimizer, X[ty], Y[ty],
                     X_min[ty], X_max[ty], Y_mean[ty], Y_scale[ty], utt_lengths[ty])


# In[40]:

torch.save(models['duration'].state_dict(), './model_durationc')







# ### Training acoustic model

# In[41]:


ty = "acoustic"
optimizer = optim.Adam(models[ty].parameters(), lr=lr, weight_decay=weight_decay)
loss_history = train_rnn(models[ty], optimizer, X[ty], Y[ty],
                     X_min[ty], X_max[ty], Y_mean[ty], Y_scale[ty], utt_lengths[ty])

torch.save(models['acoustic'].state_dict(), './model_acoustic_no_accent')




# In[ ]:
plt.figure()
plt.plot(loss_history_1["train"], linewidth=2, label="Train loss")
plt.plot(loss_history_1["test"], linewidth=2, label="Test loss")
plt.savefig('duration_loss.png')

plt.figure()
plt.plot(loss_history["train"], linewidth=2, label="Train loss")
plt.plot(loss_history["test"], linewidth=2, label="Test loss")

plt.savefig('duration_loss.png')


# ## Test
# 
# Let's see how our network works.

# ### Parameter generation utilities
# 
# Almost same as DNN text-to-speech synthesis. The difference is that we need to give initial hidden states explicitly.

# In[31]:


binary_dict, continuous_dict = hts.load_question_set("./data/questions_jp.hed")

def gen_parameters(y_predicted):
    # Number of time frames
    T = y_predicted.shape[0]
    
    # Split acoustic features
    mgc = y_predicted[:,:lf0_start_idx]
    lf0 = y_predicted[:,lf0_start_idx:vuv_start_idx]
    vuv = y_predicted[:,vuv_start_idx]
    bap = y_predicted[:,bap_start_idx:]
    
    # Perform MLPG
    ty = "acoustic"
    mgc_variances = np.tile(Y_var[ty][:lf0_start_idx], (T, 1))
    mgc = paramgen.mlpg(mgc, mgc_variances, windows)
    lf0_variances = np.tile(Y_var[ty][lf0_start_idx:vuv_start_idx], (T,1))
    lf0 = paramgen.mlpg(lf0, lf0_variances, windows)
    bap_variances = np.tile(Y_var[ty][bap_start_idx:], (T, 1))
    bap = paramgen.mlpg(bap, bap_variances, windows)
    
    return mgc, lf0, vuv, bap

def gen_waveform(y_predicted, do_postfilter=False):  
    y_predicted = trim_zeros_frames(y_predicted)
        
    # Generate parameters and split streams
    mgc, lf0, vuv, bap = gen_parameters(y_predicted)
    
    if do_postfilter:
        mgc = merlin_post_filter(mgc, alpha)
        
    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    
    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            fs, frame_period)
    return generated_waveform
    
def gen_duration(hts_labels, duration_model):
    # Linguistic features for duration
    duration_linguistic_features = fe.linguistic_features(hts_labels,
                                               binary_dict, continuous_dict,
                                               add_frame_features=False,
                                               subphone_features=None).astype(np.float32)

    # Apply normalization
    ty = "duration"
    duration_linguistic_features = minmax_scale(duration_linguistic_features, 
                                       X_min[ty], X_max[ty], feature_range=(0.01, 0.99))
    
    # Apply models
    duration_model = duration_model.cpu()
    duration_model.eval()
    
    #  Apply model
    x = Variable(torch.from_numpy(duration_linguistic_features)).float()
    try:
        duration_predicted = duration_model(x).data.numpy()
    except:
        h, c = duration_model.init_hidden(batch_size=1)
        xl = len(x)
        x = x.view(1, -1, x.size(-1))
        duration_predicted = duration_model(x, [xl], h, c).data.numpy()
        duration_predicted = duration_predicted.reshape(-1, duration_predicted.shape[-1])
                     
    # Apply denormalization
    duration_predicted = duration_predicted * Y_scale[ty] + Y_mean[ty]
    duration_predicted = np.round(duration_predicted)
    
    # Set minimum state duration to 1
    duration_predicted[duration_predicted <= 0] = 1
    hts_labels.set_durations(duration_predicted)
    
    return hts_labels    


def test_one_utt(hts_labels, duration_model, acoustic_model, post_filter=True):
    # Predict durations
    duration_modified_hts_labels = gen_duration(hts_labels, duration_model)
    
    # Linguistic features
    linguistic_features = fe.linguistic_features(duration_modified_hts_labels, 
                                                  binary_dict, continuous_dict,
                                                  add_frame_features=True,
                                                  subphone_features=acoustic_subphone_features)
    # Trim silences
    indices = duration_modified_hts_labels.silence_frame_indices()
    linguistic_features = np.delete(linguistic_features, indices, axis=0)

    # Apply normalization
    ty = "acoustic"
    linguistic_features = minmax_scale(linguistic_features, 
                                       X_min[ty], X_max[ty], feature_range=(0.01, 0.99))
    
    # Predict acoustic features
    acoustic_model = acoustic_model.cpu()
    acoustic_model.eval()
    x = Variable(torch.from_numpy(linguistic_features)).float()
    try:
        acoustic_predicted = acoustic_model(x).data.numpy()
    except:
        h, c = acoustic_model.init_hidden(batch_size=1)
        xl = len(x)
        x = x.view(1, -1, x.size(-1))
        acoustic_predicted = acoustic_model(x, [xl], h, c).data.numpy()
        acoustic_predicted = acoustic_predicted.reshape(-1, acoustic_predicted.shape[-1])
             
    # Apply denormalization
    acoustic_predicted = acoustic_predicted * Y_scale[ty] + Y_mean[ty]
    
    return gen_waveform(acoustic_predicted, post_filter)    




test_label_paths = sorted(glob(join(DATA_ROOT, "test_label_phone_align", "*.lab")))

# Save generated wav files for later comparizon
save_dir = join("./generated/jp-02-tts")
if not exists(save_dir):
    os.makedirs(save_dir)
    
for label_path, wav_path1, wav_path2 in test_label_paths:

    
    print("MyRNN")
    hts_labels = hts.load(label_path)
    waveform = test_one_utt(hts_labels, models["duration"], models["acoustic"])
    wavfile.write(join(save_dir, basename(wav_path1)), rate=fs, data=waveform)


