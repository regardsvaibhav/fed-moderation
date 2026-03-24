"""
PyTorch Dataset — uses pre-encoded indices (fast, no tokenization overhead).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class ModerationDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.encodings = torch.tensor(
            np.array(df['encoded'].tolist()), dtype=torch.long
        )
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx],
            'attention_mask': torch.ones(self.encodings.shape[1], dtype=torch.long),
            'labels': self.labels[idx]
        }


def get_dataloader(df, batch_size=32, shuffle=True):
    return DataLoader(ModerationDataset(df), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)