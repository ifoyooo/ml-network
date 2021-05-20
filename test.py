import argparse
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import pathlib
import os
from challenge_dataset import ChallengeDataset, simple_transform
from score import ChallengeMetric
import seaborn as sns

from networks import *

__author__ = "ifoyooo"
__email__ = "wangfuyun_000@foxmail.com"


'''
TEST the MSE LOSS OF THE WHOLE SET.
TEST THE SCORE OF THE WHOLE TRAINSET.
'''
project_dir = pathlib.Path(__file__).parent.absolute()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc_train_path", help="train path", type=str, default=pathlib.Path(
        __file__).parent.absolute()/"data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train")
    parser.add_argument("--lc_val_path", help="val path", type=str, default=pathlib.Path(__file__).parent.absolute() /
                        "data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train")
    parser.add_argument("--params_train_path", help="params_train_path", type=str, default=pathlib.Path(
        __file__).parent.absolute()/"data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train")
    parser.add_argument("--params_val_path", help="params_val_path", type=str, default=pathlib.Path(
        __file__).parent.absolute()/"data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train")
    parser.add_argument("--train_size", type=int, default=1256)
    parser.add_argument("--val_size", type=int, default=125600)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_from", type=int, default=10)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available()else "cpu")
    # parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intput_dim", type=int, default=55*300)
    parser.add_argument("--output_dim", type=int, default=55)
    parser.add_argument("--model", type=str, default="Conv2d")
    parser.add_argument("--MLlinearlist", type=list,
                        default=[55*300, 1024, 256])
    parser.add_argument("--continue_train", type=bool, default=False)

    args = parser.parse_args()
    model = SimpleConv().double().to(args.device)
    dataset_val = ChallengeDataset(args.lc_train_path, args.params_train_path, shuffle=True, start_ind=0,
                                   max_size=args.val_size, transform=simple_transform, device=args.device, seed=args.seed)
    valbatchsize = args.val_size//4
    loader_val = DataLoader(dataset_val, batch_size=valbatchsize)
    if args.continue_train and "model_state.pt" in os.listdir(project_dir / ('outputs/'+args.model)):
        model.load_state_dict(torch.load(
            project_dir / ('outputs/'+args.model+'/model_state.pt')))
    else:
        FileNotFoundError
    # Define Loss, metric and argsimizer
    loss_function = MSELoss()
    challenge_metric = ChallengeMetric()

    train_loss = 0
    val_loss = 0
    val_score = 0
    model.eval()
    for k, item in enumerate(loader_val):
        pred = model(item['lc'])
        loss = loss_function(item['target'], pred)
        score = challenge_metric.score(item['target'], pred)
        val_loss += loss.detach().item()
        val_score += score.detach().item()
    val_loss /= len(loader_val)
    val_score /= len(loader_val)
    print('Training loss', round(train_loss, 6))
    print('Val loss', round(val_loss, 6))
    print('Val score', round(val_score, 2))
