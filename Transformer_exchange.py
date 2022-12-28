from torch.autograd import Variable

from dataloader.Exchange import Exchange
from dataloader.JSB import JSB
from utils import set_seed_device, load_data_mat, load_data_exchange, split_seq, generate_square_subsequent_mask
from model_transformer import TransformerForecastNet
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
parser.add_argument('--lr', default=1.e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--nEpoch', default=100, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--in_features', default=1, type=int, help='input dimension')
parser.add_argument('--data', default='exchange', help='choose dataset')
parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=64, type=int, help='dimensionality of hidden layer')
parser.add_argument('--pred_len', default=1, type=int, help='prediction horizon')


opt = parser.parse_args()
opt.device = set_seed_device(opt.seed)

# -------------- generate train and test set --------------
# load data, convert to tensors and
train_data = Exchange('./data/', flag='train')
valid_data = Exchange('./data/', flag='val')
test_data = Exchange('./data/', flag='test')


train_loader = Data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
valid_loader = Data.DataLoader(dataset=valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, drop_last=True)

# -------------- initialize model & optimizer --------------
model = TransformerForecastNet(opt, opt.in_features, opt.rnn_size, nhead=2, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256).to(opt.device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4, T_0=(opt.nEpoch+1)//2, T_mult=1)
criterion = torch.nn.MSELoss()

# -------------- train & eval the model --------------
best_vloss = 1e8
total_tr_losses, total_va_losses, total_te_losses = [], [], []
for epoch in range(opt.nEpoch):
    # if epoch:
    #     scheduler.step()
    losses_train, losses_val, losses_test = [], [], []
    model.train()
    for batch_x, batch_y in train_loader:
        x, target = batch_x.to(opt.device).float(), batch_y.to(opt.device).float()
        optimizer.zero_grad()
        output = model(x, target)

        # output, target = output[:, -opt.pred_len:], target[:, -opt.pred_len:]
        loss = criterion(target, output)
        loss.backward()
        optimizer.step()

        output = train_loader.dataset.scaler.inverse_transform(output[:, [-1]].squeeze(-1).detach().cpu())
        target = train_loader.dataset.scaler.inverse_transform(target[:, [-1]].squeeze(-1).detach().cpu())
        target, output = torch.tensor(target), torch.tensor(output)
        loss = criterion(target, output)

        losses_train.append(loss.item())

    # validation iter
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            x, target = batch_x.to(opt.device).float(), batch_y.to(opt.device).float()
            output = model(x, target)

            output = valid_loader.dataset.scaler.inverse_transform(output[:, [-1]].squeeze(-1).cpu())
            target = valid_loader.dataset.scaler.inverse_transform(target[:, [-1]].squeeze(-1).cpu())
            target, output = torch.tensor(target), torch.tensor(output)
            loss = criterion(target, output)

            losses_val.append(loss.item())
        vloss = np.mean(losses_val)
        if vloss < best_vloss:
            best_vloss = vloss

    # test iter
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            x, target = batch_x.to(opt.device).float(), batch_y.to(opt.device).float()

            output = model(x, target)
            output = test_loader.dataset.scaler.inverse_transform(output[:, [-1]].squeeze(-1).cpu())
            target = test_loader.dataset.scaler.inverse_transform(target[:, [-1]].squeeze(-1).cpu())
            target, output = torch.tensor(target), torch.tensor(output)
            loss = criterion(target, output)

            losses_test.append(loss.item())

    print('[Epoch {:03d}] Train loss: {:.7f} | Valid loss: {:.7f} | Test loss: {:.7f}'.format(epoch, np.mean(losses_train), np.mean(losses_val), np.mean(losses_test)))
    total_tr_losses.append(np.mean(losses_train))
    total_va_losses.append(np.mean(losses_val))
    total_te_losses.append(np.mean(losses_test))


