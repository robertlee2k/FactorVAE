import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import datetime
import random
from tqdm.auto import tqdm

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.groups = df.groupby('datetime')
        self.rows_to_keep = []
        for _, group in self.groups:
            # drop row if their LABEL0 is NaN
            group = group.dropna(subset=['LABEL0'])
            if len(group) < 300:
                new_group = pd.DataFrame(np.nan, index=np.arange(300), columns=df.columns)
                new_group.iloc[:len(group), :] = group.values
                
                # sort by LABEL0
                new_group = new_group.sort_values(by='LABEL0', ascending=False)
                new_group = new_group.fillna(0)
                
                # ignore index
                new_group = new_group.reset_index(drop=True)
                self.rows_to_keep.append(new_group)
                assert new_group['LABEL0'].isnull().sum() == 0, 'NaN values should not be present in the dataframe'
            else:
                self.rows_to_keep.append(group)
        self.df = pd.concat(self.rows_to_keep, ignore_index=True)
        assert self.df['LABEL0'].isnull().sum() == 0, 'LABEL0 column should not contain any NaN values'
        self.input_size = 30 * len(self.df.columns)  # calculate input size based on 30 days of data

    def __len__(self):
        return len(self.df) - 30  # subtract 30 to account for accumulation of 30 days of data

    def __getitem__(self, idx):
        input_data = torch.tensor(self.df.iloc[idx:idx+30].values).float()  # accumulate 30 days of data
        label = torch.tensor(self.df.iloc[idx+30]['LABEL0']).float()  # get label for the 31st day
        label =label.unsqueeze(0)
        return input_data, label
