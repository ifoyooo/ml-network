'''
predict the validation set.
'''

from torch.utils import data
from torch.utils.data import dataset
from networks import *
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
from tqdm import tqdm


project_dir = pathlib.Path(__file__).parent.absolute()
testdir = project_dir / \
    "data/noisy_test/home/ucapats/Scratch/ml_data_challenge/test_set/noisy_test"
dataset_test = ChallengeDataset(testdir, None, shuffle=False, start_ind=0,
                                max_size=len(os.listdir(testdir)), transform=simple_transform, device="cuda"if torch.cuda.is_available() else "cpu")

loader_test = DataLoader(
    dataset_test, batch_size=64, shuffle=False)

model = SimpleConv().double().to("cuda"if torch.cuda.is_available() else "cpu")

# model=SimpleConv().double().to("cuda")
params=torch.load(
            project_dir / ('outputs/'+'Conv2d'+'/model_state.pt'))

model.load_state_dict(params)
print("load model!")
# if "model_state.pt" in os.listdir(project_dir / ('outputs/'+"Conv2d")):
    # print("load model!")
    # model.ld_state_dict(torch.load(project_dir / ('outputs/'+"Conv2d"+'/model_state.pt')))
model.eval()
print("begin:")
results=None
for key,value in tqdm(enumerate(loader_test)):
    pred=model(value["lc"])
    pred=pred.cpu().detach().numpy()
    if type(results)!=np.ndarray:
        results=pred
    else:
        results=np.vstack((results,pred))
np.savetxt("result.txt", results, fmt = '%f', delimiter ='  ')

