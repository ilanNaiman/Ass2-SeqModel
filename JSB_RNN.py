from dataloader.JSB import JSB
from utils import set_seed_device, load_data_mat, my_collate
from model import ForecastNet

import torch
import torch.optim as optim
import torch.utils.data as Data
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=5.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--nEpoch', default=100, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--in_features', default=88, type=int, help='input dimension')
parser.add_argument('--data', default='JSB', help='choose dataset')
parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--t_model', default='LSTM', type=str, help='model type')
parser.add_argument('--rnn_size', default=128, type=int, help='dimensionality of hidden layer')


opt = parser.parse_args()
opt.device = set_seed_device(opt.seed)

# -------------- generate train and test set --------------
# load data, convert to tensors and
train_set, valid_set, test_set = load_data_mat()
train_data = JSB(data=train_set)
valid_data = JSB(data=valid_set)
test_data = JSB(data=test_set)


train_loader = Data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate, drop_last=True)
valid_loader = Data.DataLoader(dataset=valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate, drop_last=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)

# -------------- initialize model & optimizer --------------
model = ForecastNet(opt.in_features, opt.rnn_size).to(opt.device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4, T_0=(opt.nEpoch+1)//2, T_mult=1)
model_name = "{0}_{1}.pt".format(opt.data, opt.t_model)

# -------------- train & eval the model --------------
best_vloss = 1e8
total_tr_losses, total_va_losses, total_te_losses = [], [], []
for epoch in range(opt.nEpoch):
    # if epoch:
    #     scheduler.step()
    losses_train, losses_val, losses_test = [], [], []
    model.train()
    for data in train_loader:
        x, target, idx, mask = data['input'], data['target'], data['idx'], data['mask'].to(opt.device)
        x, target = x.to(opt.device).float(), target.to(opt.device).float()
        optimizer.zero_grad()
        output = model(pack_padded_sequence(x, enforce_sorted=False, lengths=idx, batch_first=True), idx)
        loss = model.loss(target, output).masked_select(mask).sum() / sum(idx)
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())


    # validation iter
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            x, target, idx, mask = data['input'], data['target'], data['idx'], data['mask'].to(opt.device)
            x, target = x.to(opt.device).float(), target.to(opt.device).float()
            output = model(x)
            loss = model.loss(target, output).masked_select(mask).sum() / sum(idx)

            losses_val.append(loss.item())
        vloss = np.mean(losses_val)
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss

    # test iter
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x, target, idx, mask = data['input'], data['target'], data['idx'], data['mask'].to(opt.device)
            x, target = x.to(opt.device).float(), target.to(opt.device).float()
            output = model(pack_padded_sequence(x, enforce_sorted=False, lengths=idx, batch_first=True), idx)
            loss = model.loss(target, output).masked_select(mask).sum() / sum(idx)

            losses_test.append(loss.item())

    print('[Epoch {:03d}] Train loss: {:.4f} | Valid loss: {:.4f} | Test loss: {:.4f} | lr: {:.5f}'.
          format(epoch, np.mean(losses_train), np.mean(losses_val), np.mean(losses_test), opt.lr))
    total_tr_losses.append(np.mean(losses_train))
    total_va_losses.append(np.mean(losses_val))
    total_te_losses.append(np.mean(losses_test))


