import argparse
import torch
import numpy as np
from scipy.io import loadmat
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from model import ForecastNet
from utils import set_seed_device

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--nEpoch', default=100, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--in_features', default=1, type=int, help='input dimension')
parser.add_argument('--data', default='exchange', help='choose dataset')
parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=150, type=int, help='dimensionality of hidden layer')
parser.add_argument('--pred_len', default=1, type=int, help='prediction horizon')


opt = parser.parse_args()
opt.device = set_seed_device(opt.seed)


def JSB_data_generator():
    print('loading JSB data...')
    data = loadmat('./data/JSB_Chorales.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test


def Exchange_data_generator(seq_len=None):
    print('Exchange data...')
    data = pd.read_csv('./data/exchange_rate.csv')

    exchange_ot = data['OT']
    if seq_len is not None:
        X_train, X_test = train_test_split(exchange_ot.values, test_size=0.2, shuffle=False)
        X_train = np.array([X_train[i:i + seq_len] for i in range(0, len(X_train) - seq_len, 1)])
        X_test = np.array([X_test[i:i + seq_len] for i in range(0, len(X_test) - seq_len, 1)])
        X_train, X_valid = train_test_split(X_train, test_size=0.2, shuffle=True)
    else:
        X_train, X_test = train_test_split(exchange_ot, test_size=0.2, shuffle=False)
        X_test, X_valid = train_test_split(X_test, test_size=0.5, shuffle=False)

    X_train = torch.Tensor(X_train.astype(np.float64))
    X_valid = torch.Tensor(X_valid.astype(np.float64))
    X_test = torch.Tensor(X_test.astype(np.float64))

    return X_train, X_valid, X_test


class Exchange_splited(Dataset):

    def __init__(self, data, sacler=None, no_scaler=False):
        self.data = data
        shape = self.data.shape
        self.no_scaler = no_scaler
        if sacler is not None and not no_scaler:
            self.scaler = sacler
            self.data = self.scaler.transform(self.data.reshape(-1, 1)).reshape(shape)
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data.reshape(-1, 1))
            self.data = self.scaler.transform(self.data.reshape(-1, 1)).reshape(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Exchange(Dataset):

    def __init__(self, data, length):
        self.data = data
        self.length = length

    def __len__(self):
        return len(self.data) // self.length

    def __getitem__(self, idx):
        return self.data[idx * self.length:idx * self.length + self.length]


if __name__ == '__main__':
    X_train, X_valid, X_test = Exchange_data_generator(128)
    dataset_train = Exchange_splited(X_train)
    dataset_test = Exchange_splited(X_test, dataset_train.scaler)
    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True)

    for d in test_loader:
        d2 = dataset_train.inverse_transform(d)
        print(d2.shape)



X_train, X_valid, X_test = Exchange_data_generator(64)
train_dataset = Exchange_splited(X_train)
valid_dataset = Exchange_splited(X_valid, sacler=train_dataset.scaler)
test_dataset = Exchange_splited(X_test, sacler=train_dataset.scaler)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)


# -------------- initialize model & optimizer --------------
model = ForecastNet(opt.in_features, opt.rnn_size).to(opt.device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
criterion = torch.nn.MSELoss()

# -------------- train & eval the model --------------
best_vloss = 1e8
total_tr_losses, total_va_losses, total_te_losses = [], [], []

for epoch in range(opt.nEpoch):
    if epoch:
        scheduler.step()
    losses_train, losses_val, losses_test = [], [], []
    model.train()
    for idx, data in enumerate(train_loader, 0):
        data = data.to(opt.device)
        x, y = data[:, :-1].unsqueeze(-1), data[:, 1:].unsqueeze(-1)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(y, output)
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

    # validation iter
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(valid_loader, 0):
            data = data.to(opt.device)
            x, y = data[:, :-1].unsqueeze(-1), data[:, 1:].unsqueeze(-1)
            output = model(x).squeeze(-1).cpu()
            # output, target = output[:, -opt.pred_len:], target[:, -opt.pred_len:]
            output = valid_loader.dataset.scaler.inverse_transform(output[:, [-1]])
            y = valid_loader.dataset.scaler.inverse_transform(y[:, [-1]].squeeze(-1).cpu())
            y, output = torch.tensor(y), torch.tensor(output)
            loss = criterion(output, y)
            losses_val.append(loss.item())
        vloss = np.mean(losses_val)
        if vloss < best_vloss:
            best_vloss = vloss
        # if epoch > 10 and vloss > max(total_va_losses[-3:]):
        #     opt.lr /= 5
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = opt.lr


    # test iter
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader, 0):
            data = data.to(opt.device)
            x, y = data[:, :-1].unsqueeze(-1), data[:, 1:].unsqueeze(-1)
            output = model(x).squeeze(-1).cpu()
            # output, target = output[:, -opt.pred_len:], target[:, -opt.pred_len:]
            output = test_loader.dataset.scaler.inverse_transform(output[:, [-1]])
            y = test_loader.dataset.scaler.inverse_transform(y[:, [-1]].squeeze(-1).cpu())
            y, output = torch.tensor(y), torch.tensor(output)
            loss = criterion(output, y)

            losses_test.append(loss.item())

    print('[Epoch {:03d}] Train loss: {:.7f} | Valid loss: {:.7f} | Test loss: {:.7f}'.format(epoch, np.mean(losses_train), np.mean(losses_val), np.mean(losses_test)))
    total_tr_losses.append(np.mean(losses_train))
    total_va_losses.append(np.mean(losses_val))
    total_te_losses.append(np.mean(losses_test))


