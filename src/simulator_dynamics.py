import os
import time
import sys
import numpy as np
import argparse
import yaml

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gnn.model import DynamicsPredictor
from gnn.utils import set_seed, umeyama_algorithm
from simulator_dataset import SimDataset



def simulation_dynamics(state, action, state_mask, rope_physics_param):
    # max_nobj = 2000
    # state: (batch_size, max_nobj, 3)
    # action: (batch_size, 4)  x_start, z_start, x_end, z_end
    # state_mask: (batch_size, max_nobj, 1)
    # rope_physics_param: (batch_size, 3) in range [0, 1]
    
    # TODO implement this function
    return



def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        print(f'[{name}] epoch {cnt}')
        cnt += 1
        for data in dataloader:
            yield data

def grad_manager(phase):
    if phase == 'train':
        return torch.enable_grad()
    else:
        return torch.no_grad()

def train(config):
    train_config = config['train_config']
    model_config = config['model_config']
    dataset_config = config['dataset_config']
    material_config = config['material_config']
    
    torch.autograd.set_detect_anomaly(True)
    set_seed(train_config['random_seed'])
    device = train_config['device']
    print(f"device: {device}")

    # os.makedirs(train_config['out_dir'], exist_ok=True)
    # os.makedirs(os.path.join(train_config['out_dir'], 'checkpoints'), exist_ok=True)

    # data loader
    phases = train_config['phases']
    dataset_config['n_his'] = train_config['n_his']
    dataset_config['n_future'] = train_config['n_future']
    datasets = {phase: SimDataset(
        dataset_config=dataset_config,
        material_config=material_config,
    ) for phase in phases}
    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=train_config['batch_size'],
        shuffle=(phase == 'train'),
        num_workers=1,
    ) for phase in phases}
    dataloaders = {phase: dataloader_wrapper(dataloaders[phase], phase) for phase in phases}

    loss_plot_list_train, loss_plot_list_valid = [], []
    for epoch in range(train_config['n_epochs']):
        time1 = time.time()
        for phase in phases:
            loss_sum_list = []
            n_iters = train_config['n_iters_per_epoch'][phase] \
                    if train_config['n_iters_per_epoch'][phase] != -1 else len(datasets[phase])
            for i in range(n_iters):
                data = next(dataloaders[phase]) # graph
                
                data = {key: data[key].to(device) for key in data.keys()}

                import ipdb; ipdb.set_trace()
                pred_state = simulation_dynamics(**data)
                import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/rope_sim.yaml')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    train(config)
