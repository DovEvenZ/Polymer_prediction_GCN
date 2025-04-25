import torch
import os, shutil
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from typing import List, Optional, Callable

class CustomDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        data_file: str,
        feature_list: List[str],
        target_list: List[str],
        transform: Optional[Callable]=None,
        pre_transform: Optional[Callable]=None,
        pre_filter: Optional[Callable]=None,
        reprocess: bool=False,
        ):
        self.root = root
        self.data_file = data_file
        self.feature_list = feature_list
        self.target_list = target_list
        self.reprocess = reprocess
        if self.reprocess:
            self._reprocess()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _reprocess(self):
        if os.path.exists(os.path.join(self.root, 'processed/')):
            shutil.rmtree(os.path.join(self.root, 'processed/'))

    def process(self):
        data_list = []
        database = pd.read_csv(self.data_file).dropna(axis = 0, how = 'any')
        feature = torch.tensor(
            np.array(database.loc[:, self.feature_list]),
            dtype = torch.float
            ).reshape(-1, len(self.feature_list))
        target = torch.tensor(
            np.array(database.loc[:, self.target_list]),
            dtype = torch.float
            ).reshape(-1, len(self.target_list))
        for i in range(len(target)):
            _feature = feature[i].unsqueeze(0)
            _target = target[i].unsqueeze(0)
            data = Data(feature = _feature, y = _target)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        '''define default path to input data files.
        '''
        return ['data.csv']

    @property
    def processed_file_names(self) -> str:
        '''define default path to processed data file.
        '''
        return 'fnn_data.pt'

class CustomSubset(Subset):
    """A custom subset class that retains the 'feature' and 'y' attributes."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.feature = torch.as_tensor(
                [dataset.data.feature[i].tolist() for i in indices]
            )  # Retain the 'feature' attribute
        self.y = torch.as_tensor(
                [dataset.data.y[i].tolist() for i in indices]
            )  # Retain the 'y' attribute

    def __getitem__(self, idx):
        # Support indexing
        feature, y = self.dataset.data.feature[self.indices[idx]], self.dataset.data.y[self.indices[idx]]
        return feature, y

    def __len__(self):
        # Support len()
        return len(self.indices)
