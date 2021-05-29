from modules import *
import torch.nn as nn
import torch.nn.functional as F
import torch


# origin .
# 最起码要能够拟合训练集。降低到0.0001一下。??最起码训练误差降到0.0001一下我再说。
# 首先要完全拟合，其次需要在valset上取得0.9以上的score。
class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, [16, 2], [1, 1], padding=[
                               4, 0])  
        self.pool1 = nn.MaxPool2d([2, 1], [2, 1])  
                                        #把这里从1修改成了2
        self.conv21 = nn.Conv2d(4, 1, [16, 1], [1, 1], padding=[
                                4, 0])  
        # self.conv22 = nn.Conv2d(
            # 1, 1, [4, 1], [1, 1], padding=[0, 0])  
        self.pool2 = nn.MaxPool2d([2, 1], [1, 1])  
                                #这里从4784该陈列馆4768，考虑列之间的小的间隔，有可能能带来更好地结果。
        self.mll = nn.Sequential(nn.Linear(4784, 1024),
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


# class SimpleConv(nn.Module):
#     def __init__(self):
#         super(SimpleConv, self).__init__()

#         self.conv1 = nn.Conv2d(1, 4, [8, 4], [1, 1], padding=[
#                                4, 0])  
#         self.pool1 = nn.MaxPool2d([2, 1], [2, 1])  
#                                         #把这里从1修改成了2
#         self.conv21 = nn.Conv2d(4, 1, [16, 2], [1, 1], padding=[
#                                 4, 0])  
#         self.conv22 = nn.Conv2d(
#             1, 1, [1, 1], [1, 1], padding=[0, 0])  
#         self.pool2 = nn.MaxPool2d([2, 1], [1, 1])  
#                                 #这里从4784该陈列馆4768，考虑列之间的小的间隔，有可能能带来更好地结果。
#         self.mll = nn.Sequential(nn.Linear(5920, 1024),
#                               nn.ReLU(),
#                               nn.Linear(1024, 256),
#                               nn.ReLU(),
#                               nn.Linear(256, 55),
#                               )

#     def forward(self, input):
#         x = input.unsqueeze(1)
#         # print(x.shape)
#         x = self.conv1(x)
#         # print(x.shape)
#         x = self.pool1(x)
#         # print(x.shape)
#         x = F.relu(x)
#         x = self.conv21(x)
#         # print(x.shape)
#         x = self.conv22(x)
#         # print(x.shape)
#         x = self.pool2(x)
#         x = F.relu(x)
#         x = torch.flatten(x,start_dim=1)
#         x = self.mll(x)
#         return x

