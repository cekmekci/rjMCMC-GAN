"""
Adapted from the original content available on the Github repository:

https://github.com/ETZET/MCMC_GAN

No changes are made in this version. 
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    """
    Custom Data set for pytorch loading
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample_np = self.data[idx, :]
        sample = torch.from_numpy(sample_np)
        return sample


class MinMaxScaler():
    """
    A data scaler that scale data to given range and scale back to initial range
    """

    def __init__(self, range=(-1,1)):
        self.min = np.empty(0)
        self.max = np.empty(0)
        self.diff = np.empty(0)
        self.range = range

    def fit(self, data: np.ndarray):
        """
        Fit the scaler to data and cache the min max value of data for transform and future inversion
        :param data: np.ndarray, expected arrangement: row as instances of data and column as features of data
        """
        self.min = np.min(data, axis=0)
        max = np.max(data, axis=0)
        self.diff = max - self.min

    def transform(self, data: np.ndarray):
        """
        Transform the input data to range, using previous fitted min and max
        :param data: np.ndarray, expected arrangement: row as instances of data and column as features of data
        :param range: default to (-1,1)
        :return: data scaled to the range of input or (-1,1) by default
        """
        range_min = self.range[0]
        range_max = self.range[1]
        data_std = (data - self.min) / self.diff
        data_scaled = data_std * (range_max - range_min) + range_min
        return data_scaled

    def inverse_transform(self, data_scaled: np.ndarray):
        """
        Transform the scaled data back to original data range
        :param data_scaled: np.ndarray, expected arrangement: row as instances of data and column as features of data
        :return: data, at original scale
        """
        range_min = self.range[0]
        range_max = self.range[1]
        data_std = (data_scaled - range_min) / (range_max - range_min)
        data = data_std * self.diff + self.min
        return data


def slice_dataset(data: np.ndarray, slice_ratio = None):
    """
    partition data at different ratio cutoff, e.g. first 20%, first 40%...
    """
    if slice_ratio is None:
        slice_ratio = [0.2,0.4,0.6,0.8,1]
    length,dim = data.shape
    data_slices = []
    for ratio in slice_ratio:
        end_point = int(np.floor(ratio*length))
        data_slices.append(data[:end_point,:])
    return data_slices
