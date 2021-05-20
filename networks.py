from modules import *
import torch.nn as nn
import torch.nn.functional as F
import torch


# 最起码要能够拟合训练集。降低到0.0001一下。??最起码训练误差降到0.0001一下我再说。
# 每条curve中的连续点比连续curve要好得多。所以我们也可以从这些方面尝试 82效果挺好。
class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, [16, 1], [1, 1], padding=[
                               4, 0])  
        self.pool1 = nn.MaxPool2d([2, 1], [2, 1])  
        self.conv21 = nn.Conv2d(3, 1, [16, 1], [1, 1], padding=[
                                4, 0])  
        # self.conv22 = nn.Conv2d(
            # 1, 1, [4, 1], [1, 1], padding=[0, 0])  
        self.pool2 = nn.MaxPool2d([2, 1], [1, 1])  

        # self.conv1 = nn.Conv2d(1, 2, [8, 1], [1, 1], padding=[
        #                        4, 0])  
        # self.pool1 = nn.MaxPool2d([2, 1], [1, 1])  
        # self.conv21 = nn.Conv2d(2, 4, [4, 1], [1, 1], padding=[
        #                         4, 0])  
        # self.conv22 = nn.Conv2d(
        #     4, 1, [4, 1], [1, 1], padding=[0, 0])  
        # self.pool2 = nn.MaxPool2d([2, 1], [1, 1])  
        self.mll = nn.Sequential(nn.Linear(4800, 1024),
                              nn.ReLU(),
                              nn.Linear(1024, 256),
                              nn.ReLU(),
                              nn.Linear(256, 55),
                              )

    def forward(self, input):
        x = input.unsqueeze(1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.conv21(x)
        # print(x.shape)
        # x = self.conv22(x)
        # print(x.shape)
        x = self.pool2(x)
        x = F.relu(x)
        x = torch.flatten(x,start_dim=1)
        x = self.mll(x)
        return x
