import argparse
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam,SGD
import pathlib
import os
from challenge_dataset import ChallengeDataset,simple_transform,my_transfrom
from score  import ChallengeMetric
import seaborn as sns

from networks import *



__author__="ifoyooo"
__email__="wangfuyun_000@foxmail.com"
project_dir = pathlib.Path(__file__).parent.absolute()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc_train_path",help="train path",type=str,default=pathlib.Path(__file__).parent.absolute()/"data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train")
    parser.add_argument("--lc_val_path",help="val path",type=str,default=pathlib.Path(__file__).parent.absolute()/"data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train")
    parser.add_argument("--params_train_path",help="params_train_path",type=str,default=pathlib.Path(__file__).parent.absolute()/"data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train")
    parser.add_argument("--params_val_path",help="params_val_path",type=str,default=pathlib.Path(__file__).parent.absolute()/"data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train")
    parser.add_argument("--train_size",type=int,default=2048)
    parser.add_argument("--val_size",type=int,default=1024)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--save_from",type=int,default=3)
    parser.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available()else "cpu")
    # parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--seed",type=int,default=2048)
    parser.add_argument("--intput_dim",type=int,default=55*300)
    parser.add_argument("--output_dim",type=int,default=55)
    parser.add_argument("--model",type=str,default="Conv2d")
    parser.add_argument("--MLlinearlist",type=list,default=[55*300,1024,256])
    parser.add_argument("--continue_train",type=bool,default=False)

    args=parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)


    #创建文件夹
    if args.model not in os.listdir(project_dir/'outputs'):
        os.makedirs(project_dir/('outputs/'+args.model))
   
    # Training data
    dataset_train = ChallengeDataset(args.lc_train_path, args.params_train_path, shuffle=True, start_ind=0,
                                   max_size=args.train_size, transform=simple_transform, device=args.device)
    #第trainsize+1到2*trainsize个数据。
    # Validation data
    dataset_val = ChallengeDataset(args.lc_train_path, args.params_train_path, shuffle=True, start_ind=args.train_size,
                                 max_size=args.val_size, transform=simple_transform, device=args.device,seed=args.seed)    
    
    trainbatchsize=args.train_size//16;
    
    valbatchsize=args.val_size//8;
    loader_train = DataLoader(dataset_train, batch_size=trainbatchsize, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=valbatchsize)

    print("-----load the trainset and valset successfully!-------")

    #### 训练过程 #### 

    
    # 选择模型
    if (args.model=="MLlinear"):
        model = MLLinear(args.MLlinearlist,args.output_dim).double().to(args.device)   
    if (args.model=="Conv2d"):
        model = SimpleConv().double().to(args.device)
    if (args.model=="lstm"):
        pass
        # model = Simplelstm().double().to(args.device)


    #加载模型参数
    if args.continue_train and "model_state.pt" in os.listdir(project_dir / ('outputs/'+args.model)):
        print("continue_train!")
        model.load_state_dict(torch.load(project_dir / ('outputs/'+args.model+'/model_state.pt')))

    # Define Loss, metric and argsimizer
    loss_function = MSELoss()
    challenge_metric = ChallengeMetric()
    #优化器
    opt = Adam(model.parameters())

    # Lists to record train and val scores
    train_losses = []
    val_losses = []
    val_scores = []
    best_val_score = 0.

    for epoch in range(1, 1+args.epochs):
        print("epoch", epoch)
        train_loss = 0
        val_loss = 0
        val_score = 0
        model.train()
        #重载__get_item__函数，这样我每次得到的数据都是输入，输出。
        for k, item in enumerate(loader_train):
            
            pred = model(item['lc'])
            loss = loss_function(item['target'], pred)
            #args里里面引用了模型的参数，首先将参数清零，然后反向传播计算梯度，最后对模型参数进行更新。
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.detach().item()
        train_loss = train_loss / len(loader_train)
        #eval要在测试之前，否则的话即使不训练，
        # 也会影响参数。
        model.eval() #
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
        train_losses += [train_loss]
        val_losses += [val_loss]
        val_scores += [val_score]

        if epoch >= args.save_from and val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), project_dir / ('outputs/'+args.model+'/model_state.pt'))

    np.savetxt(project_dir /('outputs/'+args.model+'/train_losses.txt'),
               np.array(train_losses))
    np.savetxt(project_dir / ('outputs/'+args.model+'/val_losses.txt'), np.array(val_losses))
    np.savetxt(project_dir / ('outputs/'+args.model+'/val_scores.txt'), np.array(val_scores))

    # torch.save(model.load_state_dict(), project_dir / ('outputs/'+args.model+'/model_state.pt'))
    sns.set()
    fig=sns.lineplot(range(len(train_losses)),train_losses).get_figure()
    fig.savefig(project_dir / ('outputs/'+args.model+'/loss.jpg'),dpi=400)

    
    

