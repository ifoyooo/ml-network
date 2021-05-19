import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC,abstractmethod




class BaseDataset(data.Dataset,ABC):

    def __init__(self,opt):
        self.opt=opt


    #获得数据集长度        
    @abstractmethod
    def __len__(self):
        return 0
    #获得某个数据
    @abstractmethod
    def __getitem__(self, index: int):
        pass


    #修改数据选项
    @staticmethod
    def modify_commandline_option(parser,is_train):
        return parser


#返回数据处理方法
def get_transform(opt,params=None):
    pass
    #返回一个数据处理的transfrom
    

    