from abc import ABC, abstractmethod
from collections import OrderedDict
import os
from typing_extensions import ParamSpec
import torch


class BaseModel(ABC):

    #这值各种模型，参数，优化器等等。
    def __init__(self,opt):
        pass


    @staticmethod
    def modify_commandline_options(parser,is_train):
        return parser


    #三个抽象方法用于训练。
    @abstractmethod
    #输入数据
    def set_input(self,input):
        pass
    
    @abstractmethod
    # 前向传播 控制整体的。
    def forward(self):
        pass

    #前向传播，反向传播与梯度更新。
    @abstractmethod
    def optimize_parameters(self):
        pass

    #读取各种模型中的参数
    def setup(self,opt):
        pass
    def eval(self):
        pass

    #设置不可导，且输出最后结果。
    def test(self):
        pass
    def save_networks(self,epoch):
        pass
    def load_networks(self,epoch):
        pass
    def set_requires_grad(self,nets.requires_grad=False):
        pass