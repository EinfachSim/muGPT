from torch.utils.data import Dataset
import numpy as np
import torch

class BinDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        self.data = np.fromfile(path, dtype=np.uint16)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - self.seq_len) // self.seq_len

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+self.seq_len+1].astype(np.int64))
        return x, y