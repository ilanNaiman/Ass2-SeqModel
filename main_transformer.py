from torch.autograd import Variable

from dataloader.JSB import JSB
from utils import set_seed_device
from model import TransformerForecastNet
import matplotlib.pyplot as plt

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
parser.add_argument('--nEpoch', default=100, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--in_features', default=88, type=int, help='input dimension')
parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=128, type=int, help='dimensionality of hidden layer')


opt = parser.parse_args()
opt.device = set_seed_device(opt.seed)

# -------------- generate train and test set --------------
# load data, convert to tensors and
data = loadmat('./data/JSB_Chorales.mat')

input_size = 88
train_set = data['traindata'][0]
valid_set = data['validdata'][0]
test_set = data['testdata'][0]

for data in [train_set, valid_set, test_set]:
    for i in range(len(data)):
        data[i] = torch.Tensor(data[i].astype(np.float32))


def binaryMatrix(l, idx):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for jj, ss in enumerate(seq):
            if jj >= idx[i]:
                m[i].append(torch.zeros(ss.shape[0], dtype=torch.int32))
            else:
                m[i].append(torch.ones(ss.shape[0], dtype=torch.int32))
    return m


def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item['input'] for item in batch]
    padded_data = pad_sequence(data, batch_first=True)
    targets = [item['target'] for item in batch]
    padded_targets = pad_sequence(targets, batch_first=True)
    idx = [item['idx'] for item in batch]
    mask = binaryMatrix(padded_data, idx)
    mask = (torch.cat([torch.vstack(m).unsqueeze(dim=0) for m in mask])).bool()
    return {'input': padded_data, 'target': padded_targets,  'idx': idx, 'mask': mask}


def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

train_data = JSB(data=train_set)
valid_data = JSB(data=valid_set)
test_data = JSB(data=test_set)

train_loader = Data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate, drop_last=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)

# -------------- initialize model & optimizer --------------
model = TransformerForecastNet(opt.in_features, opt.rnn_size).to(opt.device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# -------------- train & eval the model --------------
total_tr_losses, total_te_losses = [], []
for epoch in range(opt.nEpoch):
    losses_train, losses_test = [], []
    model.train()
    for data in train_loader:
        x, target, idx, mask = data['input'], data['target'], data['idx'], data['mask']
        x, target = x.to(opt.device).float(), target.to(opt.device).float()

        max_len = max(idx)
        m = []
        for ii in idx:
            m.append(torch.Tensor([False for _ in range(ii)] + [True for _ in range(max_len - ii)]))
        mask_pad = torch.vstack(m)
        optimizer.zero_grad()
        output = model(x, mask_pad, generate_square_subsequent_mask(x.shape[1]))
        loss = model.loss(target, output).masked_select(mask).sum() / sum(idx)
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

    # validation iter
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x, target, idx, mask = data['input'], data['target'], data['idx'], data['mask']
            x, target = x.to(opt.device).float(), target.to(opt.device).float()
            max_len = max(idx)
            m = []
            for ii in idx:
                m.append(torch.Tensor([False for _ in range(ii)] + [True for _ in range(max_len - ii)]))
            mask_pad = torch.vstack(m)
            optimizer.zero_grad()
            output = model(x, mask_pad, generate_square_subsequent_mask(x.shape[1]))
            loss = model.loss(target, output).masked_select(mask).sum() / sum(idx)

            losses_test.append(loss.item())

    print('[Epoch {:03d}] Train loss: {:.6f} | Test loss: {:.6f}'.format(epoch, np.mean(losses_train), np.mean(losses_test)))
    total_tr_losses.append(np.mean(losses_train))
    total_te_losses.append(np.mean(losses_test))


