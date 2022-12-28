import warnings
from utils import set_seed_device

from statsmodels.tsa.arima.model import ARIMA
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
# load data, convert to tensors and
data = loadmat('./data/JSB_Chorales.mat')
Xtest = data['testdata'][0]
# -------------- initialize model & optimizer --------------
preds = []
tgt = []
loss = 0
warnings.filterwarnings("ignore")
for x in tqdm(Xtest):
    tr, te = x.T[:, :-opt.pred_len], x.T[:, -opt.pred_len:].T
    preds = [ARIMA(t, order=(4, 1, 2)).fit().forecast() for t in tr]
    preds_l = np.clip(preds, 0, 1).T
    a = np.clip(np.log(preds_l), a_min=-100, a_max=float('inf')).T
    b = np.clip(np.log(1 - preds_l), a_min=-100, a_max=float('inf')).T
    loss_sample = -np.trace(np.matmul(te, a) + np.matmul((1 - te), b))
    print(loss_sample)
    loss += loss_sample

print(loss / Xtest.size)