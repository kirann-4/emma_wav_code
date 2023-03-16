import numpy as np
import torch
import torch.nn as nn
import scipy
from scipy.io import loadmat
import librosa
from glob import glob
import pandas as pd
import os
import shutil
from random import shuffle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
from scipy.stats import pearsonr 
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

device = 'cuda'
mat_path = '/home2/data/Kirann/dataset/DataBase/AshwinHebbar/Neutral/EmaClean'
wav_path = '/home2/data/Kirann/dataset/DataBase/AshwinHebbar/Neutral/WavClean'
filenames = sorted([f[:-4] for f in os.listdir(wav_path) if f.endswith('.wav')])


wav_train_path = 'data/wav_train'
mat_train_path = 'data/mat_train'

wav_test_path = 'data/wav_test'
mat_test_path = 'data/mat_test'

wav_validation_path = 'data/wav_validation'
mat_validation_path = 'data/mat_validation'

if not os.path.exists(wav_train_path):
  os.makedirs(wav_train_path)

if not os.path.exists(wav_validation_path):
  os.makedirs(wav_validation_path)

if not os.path.exists(wav_test_path):
  os.makedirs(wav_test_path)

if not os.path.exists(mat_train_path):
  os.makedirs(mat_train_path)

if not os.path.exists(mat_validation_path):
  os.makedirs(mat_validation_path)

if not os.path.exists(mat_test_path):
  os.makedirs(mat_test_path)

test_filenames = [f for f in filenames if int(f[-1] == '0')]
validation_filenames = [f for f in filenames if int(f[-1] == '1')]
exclude_set1 = set(test_filenames)
exclude_set2 = set(validation_filenames)
main_set = set(filenames)
train_set = main_set - exclude_set1 - exclude_set2
train_filenames = list(train_set)

for filename in test_filenames:
  wav = os.path.join(wav_path, filename + '.wav')
  mat = os.path.join(mat_path, filename + '.mat')
  shutil.copy(wav, os.path.join(wav_test_path, os.path.basename(wav)))
  shutil.copy(mat, os.path.join(mat_test_path, os.path.basename(mat)))
  #folder = os.listdir(mat_test_path)
  #for i in folder:
    #print(i)
for filename in validation_filenames:
  wav = os.path.join(wav_path, filename + '.wav')
  mat = os.path.join(mat_path, filename + '.mat')
  shutil.copy(wav, os.path.join(wav_validation_path, os.path.basename(wav)))
  shutil.copy(mat, os.path.join(mat_validation_path, os.path.basename(mat)))

for filename in train_filenames:
  wav = os.path.join(wav_path, filename + '.wav')
  mat = os.path.join(mat_path, filename + '.mat')
  shutil.copy(wav, os.path.join(wav_train_path, os.path.basename(wav)))
  shutil.copy(mat, os.path.join(mat_train_path, os.path.basename(mat)))


mat_files = os.listdir('/home2/data/Kirann/emma_wav_code/data/mat_train')
points = []


for i, mat_file in enumerate(mat_files):
    m = scipy.io.loadmat(os.path.join('/home2/data/Kirann/emma_wav_code/data/mat_train', mat_file))
    ema_data = m['EmaData']
    ema = torch.tensor(ema_data)
    indices = torch.tensor([12,13,14,15,16,17])
    target = torch.index_select(ema, 0, indices)
    hulls = []
    for j in range(3):
        x = target[2*j, :]
        y = target[2*j + 1, :]
        points.append(np.vstack((x,y)).T)      

points = np.vstack(points)
hull =  ConvexHull(points)
x_all = []
y_all = []
for simplex in hull.simplices:
    x_c = points[simplex,0]
    y_c = points[simplex,1]
    index = np.where((y_c>1) & (x_c>-48))
    x_c_f = x_c[index]           
    y_c_f = y_c[index]
    #plt.plot(x_c_f, y_c_f, 'r-', lw = 2) 

    for x, y in zip(x_c_f, y_c_f):
        if x not in x_all:
            x_all.append(x)
            y_all.append(y)           

sorted_indices = np.argsort(x_all)
x_all = np.array(x_all)[sorted_indices]
y_all = np.array(y_all)[sorted_indices]

class CustDat(Dataset):
  
  def __init__(self, mat_path, wav_path, sr = 16000, window_size = 1024, hop_length = 160, n_mfcc = 13):
    self.mat_path = mat_path
    self.wav_path = wav_path
    self.sr = sr
    self.window_size = window_size
    self.hop_length = hop_length    
    self.n_mfcc = n_mfcc
    self.filenames = sorted([f[:-4] for f in os.listdir(wav_path) if f.endswith('.wav')])
    self.wav_path = Path(wav_path)
    self.mat_path = Path(mat_path)
   
  
  def __len__(self):
    
    return len(self.filenames)
  
  def __getitem__(self, idx):
    
    filename = self.filenames[idx]
    wav_file = self.wav_path / (filename + '.wav')
    raw_data, sr = librosa.load(wav_file, sr=self.sr)
    mat_file = self.mat_path / (filename + '.mat')
    target_data = loadmat(mat_file)['EmaData']
    
    mfcc = librosa.feature.mfcc(y = raw_data, sr = self.sr, n_fft = self.window_size, hop_length = self.hop_length, n_mfcc = self.n_mfcc)
    #target = torch.tensor(target_data)
    target = torch.from_numpy(target_data)
        
    

    n_ema = target.shape[1]
    n_mfcc = mfcc.shape[1]
    if n_mfcc < n_ema:
       mfcc_padded = np.zeros((13, n_ema))
       mfcc_padded[:, :n_mfcc] = mfcc
       mfcc = mfcc_padded
    else:
       mfcc = mfcc[:, :n_ema]

    #mfcc = torch.tensor(mfcc.T)
    mfcc = torch.from_numpy(mfcc.T)
    indices = torch.tensor([0,1,2,3,8,9,12,13,14,15,16,17])
    target = torch.index_select(target, 0, indices)
    
    target_mean = torch.mean(target, axis = 1)
    target_std = torch.std(target, axis=1)
    for i in range (12):
      target[i] = (target[i]-target_mean[i]) / target_std[i]

    target_val = target.T
    return target_val, mfcc

#CustDat(mat_train_path, wav_train_path).__getitem__(0)

class CollateFn:
  def __call__(sel, batch):
    emas, mfccs = zip(*batch)
    max_frames_emas = max([e.shape[0] for e in emas])
    max_frames_mfccs = max([mfcc.shape[0] for mfcc in mfccs])
    ema_pad = torch.zeros(len(emas), max_frames_emas, emas[0].shape[1])
    mfcc_pad = torch.zeros(len(mfccs), max_frames_mfccs, mfccs[0].shape[1])
    for i, ema in enumerate(emas):
      ema_pad[i, :ema.shape[0],:] = ema
    for i, mfcc in enumerate(mfccs):  
      mfcc_pad[i, :mfcc.shape[0],:] = mfcc

    return ema_pad.to(device), mfcc_pad.to(device)

train_dl = DataLoader(CustDat(mat_train_path, wav_train_path),collate_fn = CollateFn(), batch_size = 16, shuffle = False)
val_dl = DataLoader(CustDat(mat_validation_path, wav_validation_path),collate_fn = CollateFn(), batch_size = 16, shuffle = False)
test_dl = DataLoader(CustDat(mat_test_path, wav_test_path), collate_fn = CollateFn(), batch_size = 16, shuffle = False)


CH = [-45.23562205, -42.21695848, -40.00809952, -25.05182801, -17.31076902, -3.79376159, -0.38335141, 16.29013896, 18.67840768, 20.18741574, 21.64881282, 18.64411811, 9.12052052, 1.45907601]
mean = sum(CH) / len(CH)
std = (sum([(x - mean) ** 2 for x in CH]) / len(CH)) ** 0.5
CH_std = [(x - mean) / std for x in CH]
convex_hull = np.tile(CH_std, (16,1,1))
convex_hull = convex_hull.reshape((16,14))
convex_hull = torch.tensor(convex_hull, dtype = torch.float32).cuda()



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
  
        self.layer1 = nn.Conv1d(in_channels = 13, out_channels = 32, kernel_size = 3, padding = 1)
        self.layer2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.layer3 = nn.Conv1d(in_channels = 64, out_channels = 256, kernel_size = 3, padding = 1)
        self.layer4 = nn.Conv1d(in_channels = 256, out_channels = 12, kernel_size = 3, padding = 1)
        self.layer5 = nn.Linear(in_features=256, out_features=14)
        self.relu = torch.nn.ReLU()
        
             

    def forward(self, x):
      x = x.permute(0,2,1)
      x = self.layer1(x)
      x = self.relu(x)
      x = self.layer2(x)
      x = self.relu(x)
      x = self.layer3(x)
      x = self.relu(x)
      emma = self.layer4(x)
      x = torch.mean(x, dim=2)
      x = self.layer5(x)
      emma = emma.permute(0, 2, 1)
      return emma, x

model = Net()
model = model.to(device, dtype = torch.float32)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
epochs = 50
mean_train_pearsonr = 0
mean_val_pearsonr = 0 
mean_test_pearsonr = 0
mean_co_train = []
mean_co_val = []
mean_co_test = []
TL = []
VL = []
con_pred = []


for epoch in range (epochs):
  model.train()
  train_emma_loss = 0
  train_convex_hull_loss = 0
  for emma, mfcc in train_dl:
    model = model.float()
    emma = emma.float().to(device)
    mfcc = mfcc.float().to(device)
    optimizer.zero_grad()
    emma_pred, convex_hull_pred = model(mfcc)
    con_pred.append(convex_hull_pred)

    train_emma_loss = loss(emma_pred, emma)
    train_convex_hull_loss = loss(convex_hull_pred, convex_hull)
    t_loss = train_emma_loss + train_convex_hull_loss
    pred_ema = emma_pred.detach().cpu()
    true_ema = emma.detach().cpu()
    correlations = []
    for i in range(pred_ema.shape[2]):
      corr,_ = pearsonr(true_ema[0,:,i], pred_ema[0,:,i])
      correlations.append(corr)
    mean_corr = np.mean(correlations)
    mean_co_train.append(mean_corr)  
    t_loss.float()
    t_loss.backward()
    optimizer.step()
    train_emma_loss += train_emma_loss.item()
    train_convex_hull_loss += train_convex_hull_loss.item()


  model.eval()  
  val_emma_loss = 0
  val_convex_hull_loss = 0
  with torch.no_grad():
     for emma, mfcc in train_dl:
        model = model.float()
        emma = emma.float().to(device)
        mfcc = mfcc.float().to(device)
        optimizer.zero_grad()
        emma_pred, convex_hull_pred = model(mfcc)

        val_emma_loss = loss(emma_pred, emma)
        val_convex_hull_loss = loss(convex_hull_pred, convex_hull)
        t_loss = val_emma_loss + val_convex_hull_loss
        pred_ema = emma_pred.detach().cpu()
        true_ema = emma.detach().cpu()
        correlations = []
        for i in range(pred_ema.shape[2]):
            corr,_ = pearsonr(true_ema[0,:,i], pred_ema[0,:,i])
            correlations.append(corr)
        mean_corr = np.mean(correlations)
        mean_co_val.append(mean_corr)  
        val_emma_loss += val_emma_loss.item()
        val_convex_hull_loss += val_convex_hull_loss.item()

  test_emma_loss = 0
  test_convex_hull_loss = 0
  with torch.no_grad():
     for emma, mfcc in train_dl:
        model = model.float()
        emma = emma.float().to(device)
        mfcc = mfcc.float().to(device)
        optimizer.zero_grad()
        emma_pred, convex_hull_pred = model(mfcc)

        test_emma_loss = loss(emma_pred, emma)
        test_convex_hull_loss = loss(convex_hull_pred, convex_hull)
        t_loss = test_emma_loss + test_convex_hull_loss
        pred_ema = emma_pred.detach().cpu()
        true_ema = emma.detach().cpu()
        correlations = []
        for i in range(pred_ema.shape[2]):
            corr,_ = pearsonr(true_ema[0,:,i], pred_ema[0,:,i])
            correlations.append(corr)
        mean_corr = np.mean(correlations)
        mean_co_test.append(mean_corr)  
        test_emma_loss += test_emma_loss.item()
        test_convex_hull_loss += test_convex_hull_loss.item()
    
  
  mean_train_pearsonr = np.mean(mean_co_train)   
  mean_val_pearsonr = np.mean(mean_co_val)  
  mean_test_pearsonr = np.mean(mean_co_test)  
  
  conv_pred = torch.stack(con_pred)
  av_conv_pred = torch.mean(conv_pred, dim = 0)
  print('Epoch [{}/{}], emma_train_loss:{:.4f}, emma_train_correlation_coefficient:{:.4f}, Convex_hull_train_loss:{:.4f}, emma_validation_loss:{:.4f}, emma_validation_correlation_coefficient:{:.4f}, Convex_hull_val_loss:{:.4f}, emma_test_loss:{:.4f}, emma_test_correlation_coefficient:{:.4f}, Convex_hull_test_loss:{:.4f}'.format(epoch+1, epochs, train_emma_loss, mean_train_pearsonr, train_convex_hull_loss, val_emma_loss, mean_val_pearsonr, val_convex_hull_loss, test_emma_loss, mean_test_pearsonr, test_convex_hull_loss))
  '''with torch.no_grad():
     convex_hull_pred_mean = torch.mean(av_conv_pred, dim=0).numpy()
     convex_hull_pred_mean_rescaled = (convex_hull_pred_mean * std) + mean  
     plt.plot(CH, label='Original')
     plt.plot(convex_hull_pred_mean_rescaled, label='Predicted')
     plt.legend()
     plt.show()
 
  TL.append(train_emma_loss)
  VL.append(val_emma_loss)

print(len(TL))
with torch.no_grad():
   ep = range(1, epochs+1)
   plt.plot(ep, TL, color = 'r', label = 'train_loss' )
   plt.plot(ep, VL, color = 'g', label = 'val_loss' )
   plt.xlabel('Epoch')
   plt.ylabel('train-val loss')
   plt.legend()
   plt.show()'''

