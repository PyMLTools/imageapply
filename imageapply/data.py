from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import torch
from typing import TypeVar

T = TypeVar("T", np.ndarray, torch.Tensor)

class data_object(ABC):

    def __init__(self, data):
        self.data = data

    @abstractproperty
    def shape(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def split(self, n_splits, axis):
        pass

    @abstractmethod
    def concatenate(self, axis):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

class numpy_data(data_object):

    def __init__(self, data):
        super().__init__(data)

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return numpy_data(self.data.copy())

    def split(self, n_splits, axis):
        return numpy_data(np.split(self.data, n_splits, axis=axis))

    def concatenate(self, axis):
        return numpy_data(np.concatenate(self.data, axis=axis))

    def __getitem__(self, key):
        return numpy_data(self.data[key])