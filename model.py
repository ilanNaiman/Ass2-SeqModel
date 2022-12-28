import torch
import torch.nn as nn
import torch.nn.functional as F
from tcn import TemporalConvNet
import math

from utils import split_seq, generate_square_subsequent_mask, generate_pad_mask


class ForecastNet(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(ForecastNet, self).__init__()
        self.input_dim = in_features
        self.rnn = nn.LSTM(input_size=in_features, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, in_features)
        self.loss_f = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, idx=None):
        output, _ = self.rnn(x)
        if idx:
            output, output_lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=max(idx))
        output = self.linear(output)

        return output

    def loss(self, target, output):
        return self.loss_f(output, target)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.transpose(pe, 1, 0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerForecastNet(nn.Module):
    def __init__(self, opt, in_features, hidden_size, d_model=512, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dropout=.1, dim_feedforward=256):
        super(TransformerForecastNet, self).__init__()
        self.opt = opt
        self.input_dim = in_features
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                 dim_feedforward=dim_feedforward, dropout=dropout, activation="relu", batch_first=True)
        self.positional_encoding = PositionalEncoding(d_model=hidden_size, dropout=0.1, max_len=200)
        self.linear = nn.Linear(in_features, hidden_size)
        self.linear_out = nn.Linear(hidden_size, in_features)
        self.loss_f = nn.BCEWithLogitsLoss(reduction='none')

    # def forward(self, src, mask_pad, mask_future):
    #     pe_src = self.positional_encoding(src)
    #     output = self.transformer(pe_src,
    #                               pe_src,
    #                               src_mask=mask_future,
    #                               tgt_mask=mask_future,
    #                               src_key_padding_mask=mask_pad,
    #                               tgt_key_padding_mask=mask_pad)
    #
    #     return output

    def forward(self, x, target, idx=None, pad=False):
        if not idx:
            idx = self.opt.batch_size * [x.shape[1]]

        src = self.linear(x)
        pe_src = self.positional_encoding(src)

        x_src, x_tgt, tgt, src_len, tgt_len = split_seq(pe_src, target, idx, pad=pad)
        src_mask = generate_square_subsequent_mask(dim1=x_src.shape[1], dim2=x_src.shape[1]).to(self.opt.device)
        tgt_mask = generate_square_subsequent_mask(dim1=x_tgt.shape[1], dim2=x_tgt.shape[1]).to(self.opt.device)

        if pad:
            src_mask_pad = generate_pad_mask(src_len).to(self.opt.device)
            tgt_mask_pad = generate_pad_mask(tgt_len).to(self.opt.device).to(self.opt.device)
        else:
            src_mask_pad, tgt_mask_pad = None, None

        output = self.transformer(x_src,
                                  x_tgt,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_mask_pad,
                                  tgt_key_padding_mask=tgt_mask_pad)
        output = self.linear_out(output)

        return output, x_src, x_tgt, tgt, src_len, tgt_len

    def loss(self, target, output):
        return self.loss_f(output, target)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()
        self.loss_f = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output).double()
        return output

    def loss(self, target, output):
        return self.loss_f(output, target)