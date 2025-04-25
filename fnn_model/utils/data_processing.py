from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from typing import Dict

import torch
import os, shutil
import torch_geometric.transforms as T
import numpy as np

from datasets.datasets import CustomDataset, CustomSubset
from utils.calc_mean_std import calc_mean_std

class data_processing():
    def __init__(self, param: Dict, reprocess: bool = True) -> None:
        self.path = param['path']
        self.data_file = param['data_file']
        self.feature_list = param['feature_list']
        self.target_list = param['target_list']
        self.transform = param['target_transform']
        self.seed = param['seed']
        self.split_method = param['split_method']
        self.split_path = param['SPLIT_file']
        self.train_size = param['train_size']
        self.val_size = param['val_size']
        self.batch_size = param['batch_size']
        self.reprocess = reprocess
        self.dataset = self.gen_dataset()
        self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset()
        self.train_loader, self.val_loader, self.test_loader = self.gen_loader()

    def gen_dataset(self):
        dataset = CustomDataset(
            root = self.path,
            data_file = self.data_file,
            feature_list = self.feature_list,
            target_list = self.target_list,
            reprocess = self.reprocess
            )

        dataset = self.target_transform(dataset)

        return dataset

    def target_transform(self, dataset):
        if self.transform == 'LN':
            dataset.data.y = torch.log(dataset.y)
        elif self.transform == 'LG':
            dataset.data.y = torch.log10(dataset.y)
        elif self.transform == 'E^-x':
            dataset.data.y = torch.exp(-dataset.y)
        elif not self.transform:
            pass

        return dataset

    def split_dataset(self):
        if self.split_method == 'random':
            train_size = int(self.train_size * len(self.dataset))
            val_size = int(self.val_size * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed)
                )
            train_dataset = CustomSubset(self.dataset, train_dataset.indices)
            mean, std = calc_mean_std(
                torch.cat([
                    train_dataset.feature,
                    train_dataset.y
                    ], dim=1)
                )
            dataset_scaled = (torch.cat([self.dataset.data.feature, self.dataset.data.y], dim = 1) - mean) / std
            self.dataset.data.feature = dataset_scaled[:, 0:len(self.dataset.feature_list)]
            self.dataset.data.y = dataset_scaled[:, len(self.dataset.feature_list):len(self.dataset.feature_list)+len(self.dataset.target_list)]
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed)
                )
        elif self.split_method == 'manual':
            train_dataset = self.dataset[np.load(self.split_path, allow_pickle=True)[0]]
            mean, std = calc_mean_std(
                torch.cat([
                    train_dataset.feature,
                    train_dataset.y
                    ], dim=1)
                )
            dataset_scaled = (torch.cat([self.dataset.data.feature, self.dataset.data.y], dim = 1) - mean) / std
            self.dataset.data.feature = dataset_scaled[:, 0:len(self.dataset.feature_list)]
            self.dataset.data.y = dataset_scaled[:, len(self.dataset.feature_list):len(self.dataset.feature_list)+len(self.dataset.target_list)]

            train_dataset = self.dataset[np.load(self.split_path, allow_pickle=True)[0]]
            val_dataset = self.dataset[np.load(self.split_path, allow_pickle=True)[1]]
            test_dataset = self.dataset[np.load(self.split_path, allow_pickle=True)[2]]
        else:
            raise NotImplementedError("Split method not implemented.")

        self.mean = mean
        self.std = std

        return train_dataset, val_dataset, test_dataset

    def gen_loader(self):
        train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        val_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)
        test_loader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)

        return train_loader, val_loader, test_loader

