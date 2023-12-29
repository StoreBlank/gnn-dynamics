import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from config.base_config import gen_args
from dataset.dataset_carrots import CarrotsDynDataset
from train_carrots import dataloader_wrapper
from gnn.utils import set_seed

args = gen_args()
set_seed(args.random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

phases = ['train', 'valid']

data_dirs = args.data_dir
prep_save_dir = args.prep_save_dir

n_his = 4
n_future = 3
dist_thresh = 0.05
adj_thresh_range = [0.09, 0.11]

data_kwargs = {"train": {
            "n_his": n_his,
            "n_future": n_future,
            "dist_thresh": dist_thresh,
            "adj_thresh_range": adj_thresh_range, # adj_thresh_range, # construct edge
            "fps_radius_range": adj_thresh_range, # fps sampling: determine number of nodes
            "max_n": 1, # number of objects
            "max_nobj": 100, # number of particles per object
            "max_neef": 1, # number of eef particles
            "max_nR": 500, # number of relations
            "ratios": [0, 0.9], # train/valid split
            "phys_noise": 0.01,
        },
        "valid": {
            "n_his": n_his,
            "n_future": n_future,
            "dist_thresh": dist_thresh,
            "adj_thresh_range": adj_thresh_range,
            "fps_radius_range": adj_thresh_range,
            "max_n": 1,
            "max_nobj": 100,
            "max_neef": 1,
            "max_nR": 500,
            "ratios": [0.9, 1],
            "phys_noise": 0.0,
        }
    }

batch_size = 64
datasets = {phase: CarrotsDynDataset(data_dirs, prep_save_dir, phase, **data_kwargs[phase]) for phase in phases}
dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=1,
    ) for phase in phases}
# print('train:', len(dataloaders['train']), 'valid:', len(dataloaders['valid']))\
dataloaders = {phase: dataloader_wrapper(dataloaders[phase], phase) for phase in phases}

data = next(dataloaders[phases[0]])

data = {key: data[key].to(device) for key in data.keys()}
print(data['state'])