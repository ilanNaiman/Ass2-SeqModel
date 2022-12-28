import numpy as np
from torch.utils.data import Dataset


class JSB(Dataset):
    def __init__(self, data):
        self.data = data
        self.N = self.data.shape[0]

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data = self.data[index][:-1]  # (8, 64, 64, 3)
        label = self.data[index][1:]
        idx = data.shape[0]

        return {"input": data, "target": label, "idx": idx}
