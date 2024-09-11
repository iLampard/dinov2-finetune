from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import torch
from torch.utils.data import Dataset
from registrable import Registrable


class BaseDataset(Registrable, ABC):
    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass
