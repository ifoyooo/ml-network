import argparse
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import pathlib
import os
from challenge_dataset import ChallengeDataset,simple_transform
from score  import ChallengeMetric


project_dir = pathlib.Path(__file__).parent.absolute()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc_train_path",help="train path",type=str,default=pathlib.Path(__file__).parent.absolute()/"data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train")
    parser.add_argument("--lc_val_path",help="val path",type=str,default=pathlib.Path(__file__).parent.absolute()/"data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train")
    parser.add_argument("--params_train_path",help="params_train_path",type=str,default=pathlib.Path(__file__).parent.absolute()/"data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train")
    parser.add_argument("--params_val_path",help="params_val_path",type=str,default=pathlib.Path(__file__).parent.absolute()/"data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train")
    parser.add_argument("--train_size",type=int,default=512)
    parser.add_argument("--val_size",type=int,default=512)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--save_from",type=int,default=10)
    # parser.add_argument("--device",type=str,default="cpu" if torch.cuda.is_available()else "cuda")
    parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--seed",type=int,default=0)

    # 读取数据集完成。
    opt=parser.parse_args()
    # Training
    dataset_train = ChallengeDataset(opt.lc_train_path, opt.params_train_path, shuffle=True, start_ind=0,
                                   max_size=opt.train_size, transform=simple_transform, device=opt.device)
    #第trainsize+1到2*trainsize个数据。
    # Validation
    dataset_val = ChallengeDataset(opt.lc_train_path, opt.params_train_path, shuffle=True, start_ind=opt.train_size,
                                 max_size=opt.val_size, transform=simple_transform, device=opt.device,seed=opt.seed)    
    loader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=opt.batch_size)

    print("-----load the trainset and valset successfully!-------")

    #### 训练过程 #### 

