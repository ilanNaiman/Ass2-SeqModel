import warnings
from dataloader.Exchange import Exchange
from utils import set_seed_device
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

import torch.utils.data as Data
import argparse
from tqdm import tqdm
import numpy as np


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
test_data = Exchange('./data/', flag='test')
test_loader = Data.DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=True)

# -------------- initialize model & optimizer --------------
preds = []
tgt = []
warnings.filterwarnings("ignore")
for x, y in tqdm(test_loader):
    tr, te = x[0, :-opt.pred_len], x[0, -opt.pred_len:]
    model = ARIMA(np.array(tr), order=(6,2,4))
    model_fit = model.fit()
    output = model_fit.forecast()
    preds.append(output[0])
    tgt.append(te[0].item())

rmse = mean_squared_error(test_loader.dataset.scaler.inverse_transform(np.expand_dims(preds, axis=-1)),
                          test_loader.dataset.scaler.inverse_transform(np.expand_dims(tgt, axis=-1)))
print('Test RMSE: %.7f' % rmse)
plt.plot(tgt)
plt.plot(preds, color='red')
plt.show()