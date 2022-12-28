import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import split_seq, generate_square_subsequent_mask


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
        self.positional_encoding = PositionalEncoding(d_model=hidden_size, dropout=0.1, max_len=384)
        self.linear = nn.Linear(in_features, hidden_size)
        self.linear_out = nn.Linear(hidden_size, in_features)

    def forward(self, x, target):

        x = self.linear(x)
        # pe_x = self.positional_encoding(x)
        # x_src, x_tgt, tgt, _, _ = split_seq(pe_x, target, self.opt.batch_size * [x.shape[1]], pad=False)
        src_mask = generate_square_subsequent_mask(dim1=x.shape[1], dim2=x.shape[1]).to(self.opt.device)
        # tgt_mask = generate_square_subsequent_mask(dim1=x_tgt.shape[1], dim2=x_tgt.shape[1]).to(self.opt.device)
        # mem_mask = generate_square_subsequent_mask(dim1=x_tgt.shape[1], dim2=x_src.shape[1]).to(self.opt.device)

        output = self.transformer(x,
                                  x,
                                  src_mask=src_mask,
                                  tgt_mask=src_mask,
                                  memory_mask=src_mask)
        output = self.linear_out(output)

        return output