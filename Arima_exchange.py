import warnings

from torch.autograd import Variable

from dataloader.Exchange import Exchange
from dataloader.JSB import JSB
from utils import set_seed_device, load_data_mat, load_data_exchange
from model import ForecastNet
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

import torch
import torch.optim as optim
import torch.utils.data as Data
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence
import argparse
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=5.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--in_features', default=1, type=int, help='input dimension')
parser.add_argument('--data', default='exchange', help='choose dataset')
parser.add_argument('--pred_len', default=1, type=int, help='prediction horizon')


opt = parser.parse_args()
opt.device = set_seed_device(opt.seed)

# -------------- generate train and test set --------------
# train_data = Exchange('./data/', flag='train')
# valid_data = Exchange('./data/', flag='val')
test_data = Exchange('./data/', flag='test')

test_loader = Data.DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=True)

# -------------- initialize model & optimizer --------------
preds = []
tgt = []
warnings.filterwarnings("ignore")
for x, y in tqdm(test_loader):
    tr, te = x[0, :-opt.pred_len], x[0, -opt.pred_len:]
    # plt.plot(tr.flatten())
    # plt.show()
    model = ARIMA(np.array(tr), order=(6,2,4))
    model_fit = model.fit()
    output = model_fit.forecast()
    preds.append(output[0])
    tgt.append(te[0].item())

rmse = mean_squared_error(preds, tgt)
print('Test RMSE: %.3f' % rmse)
plt.plot(tgt)
plt.plot(preds, color='red')
plt.show()