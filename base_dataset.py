import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC,abstractmethod

class BaseDataset(data.Dataset,ABC):

    def __init__(self):
        pass


    #获得数据集长度        
    @abstractmethod
    def __len__(self):
        return 0
    #获得某个数据
    @abstractmethod
    def __getitem__(self, index: int):
        pass
    