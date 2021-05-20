'''
完成特有的数据集类别
'''

from base_dataset import BaseDataset
import os
import numpy as np 
import torch
from pathlib import Path


# 获取数据集
# 通过isTrain来判断到底是训练集还是测试集
class ChallengeDataset(BaseDataset):
    def __init__(self,lc_path, params_path=None, transform=None, start_ind=0,
                 max_size=int(1e9), shuffle=True, seed=None, device=None):
        BaseDataset.__init__(self)
        self.lc_path = lc_path # 数据所在的文件夹
        self.transform = transform # 数据的处理方法
        self.device = device # 设备
        self.files = sorted(
            [p for p in os.listdir(self.lc_path) if p.endswith('txt')]) # 文件夹下的文件名
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.files)
        self.files = self.files[start_ind:start_ind+max_size] # 选择部分，并且shuffle

        if params_path is not None:
            self.params_path = params_path   # 如果有标签，那么读取标签。
        else:
            self.params_path = None   # 否则没有标签，设为none。
            self.params_files = None


    def __getitem__(self, idx: int):
        item_lc_path = Path(self.lc_path) / self.files[idx]
        lc = torch.from_numpy(np.loadtxt(item_lc_path))
        if self.transform:
            lc = self.transform(lc)
        if self.params_path is not None:
            item_params_path = Path(self.params_path) / self.files[idx]
            target = torch.from_numpy(np.loadtxt(item_params_path))
        else:
            target = torch.Tensor()
        #lc 表示模型的输入，从noisy_train中可以得到。
        #target表示模型的输出，从params_train中可以得到。
        return {'lc': lc.to(self.device),
                'target': target.to(self.device)} # 返回输入与标签。
    def __len__(self):
        return len(self.files)


#更好的预处理化和更优的模型参数。
def simple_transform(x):
    """Perform a simple preprocessing of the input light curve array
    Args:
        x: np.array
            first dimension is time, at least 30 timesteps
    Return:
        preprocessed array
    """
    out = x.clone()
    out[:,:30]=0
    # centering
    out -= 1.
    # rough rescaling
    out /= 0.04
    return out