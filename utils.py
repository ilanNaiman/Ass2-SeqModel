import torch
import numpy as np
from scipy.io import loadmat
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('cuda available')
    else:
        device = torch.device("cpu")
    return device


def load_data_mat():
    # load data, convert to tensors and
    data = loadmat('./data/JSB_Chorales.mat')

    input_size = 88
    Xtrain = data['traindata'][0]
    Xvalid = data['validdata'][0]
    Xtest = data['testdata'][0]

    for data in [Xtrain, Xvalid, Xtest]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float32))

    return Xtrain, Xvalid, Xtest


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


def generate_square_subsequent_mask(dim1, dim2):
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def split_seq(x, target, idx, pad=True):
    x_src, x_tgt, tgt = [], [], []
    src_len, tgt_len = [], []
    for xx, tt, jj in zip(x, target, idx):
        x_src.append(xx[:int(jj / 2) + 1])  # x1, ..., x_{(T-1)/2}
        x_tgt.append(xx[int(jj / 2):jj])  # x_{(T-1)/2}, ..., x_{T-1}
        tgt.append(tt[int(jj / 2):jj])  # x_{(T-1)/2 + 1}, ..., x_{T}
        src_len.append(x_src[-1].shape[0])
        tgt_len.append(x_tgt[-1].shape[0])
    if pad:
        return pad_sequence(x_src, batch_first=True),\
               pad_sequence(x_tgt, batch_first=True),\
               pad_sequence(tgt, batch_first=True), \
               src_len,\
               tgt_len
    else:
        return torch.cat([xx.unsqueeze(dim=0) for xx in x_src]),\
               torch.cat([xx.unsqueeze(dim=0) for xx in x_tgt]),\
               torch.cat([xx.unsqueeze(dim=0) for xx in tgt]), \
               src_len,\
               tgt_len



def generate_pad_mask(seq_lens):
    max_len = max(seq_lens)
    m = []
    for ii in seq_lens:
        m.append(torch.Tensor([False for _ in range(ii)] + [True for _ in range(max_len - ii)]))
    return torch.vstack(m).bool()

def plt_all_models_loss(total_tr_losses, total_va_losses, total_te_losses,
                        total_tr_losses_tcn, total_va_losses_tcn, total_te_losses_tcn,
                        total_tr_losses_trans, total_va_losses_trans, total_te_losses_trans):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    gs = fig.add_gridspec(4, 4)
    ax1 = plt.subplot(gs[:2, :2])
    ax1.plot(range(len(total_tr_losses)), total_tr_losses, label='Train')
    ax1.plot(range(len(total_va_losses)), total_va_losses, label='Val')
    ax1.plot(range(len(total_te_losses)), total_te_losses, label='Test')
    ax1.set_title('RNN - Train, Val, Test Losses')
    ax1.set_yscale('log')
    ax1.legend()
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.plot(range(len(total_tr_losses_tcn)), total_tr_losses_tcn, label='Train')
    ax2.plot(range(len(total_va_losses_tcn)), total_va_losses_tcn, label='Val')
    ax2.plot(range(len(total_te_losses_tcn)), total_te_losses_tcn, label='Test')
    ax2.set_title('TCN - Train, Val, Test Losses')
    ax2.set_yscale('log')
    ax2.legend()
    ax3 = plt.subplot(gs[2:4, 1:3])
    ax3.plot(range(len(total_tr_losses_trans)), total_tr_losses_trans, label='Train')
    ax3.plot(range(len(total_va_losses_trans)), total_va_losses_trans, label='Val')
    ax3.plot(range(len(total_te_losses_trans)), total_te_losses_trans, label='Test')
    ax3.set_title('Transformer - Train, Val, Test Losses')
    ax3.set_yscale('log')
    ax3.legend()

    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='NLL')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(f'./results/Exchange_all_train_val_test_net.pdf', transparent=True, bbox_inches='tight', pad_inches=0,
                dpi=300)
    plt.show()