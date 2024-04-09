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
from dataset import DynDataset, construct_edges_from_states

def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        print(f'[{name}] epoch {cnt}')
        cnt += 1
        for data in dataloader:
            yield data

def rigid_loss(orig_pos, pred_pos, obj_mask):
    _, R_pred, t_pred = umeyama_algorithm(orig_pos, pred_pos, obj_mask, fixed_scale=True)
    pred_pos_ume = orig_pos.bmm(R_pred.transpose(1, 2)) + t_pred
    pred_pos_ume = pred_pos_ume.detach()
    loss = F.mse_loss(pred_pos[obj_mask], pred_pos_ume[obj_mask])
    return loss

def grad_manager(phase):
    if phase == 'train':
        return torch.enable_grad()
    else:
        return torch.no_grad()

def truncate_graph(data):
    Rr = data['Rr']
    Rs = data['Rs']
    Rr_nonzero = torch.sum(Rr, dim=-1) > 0
    Rs_nonzero = torch.sum(Rs, dim=-1) > 0
    n_Rr = torch.max(Rr_nonzero.sum(1), dim=0)[0].item()
    n_Rs = torch.max(Rs_nonzero.sum(1), dim=0)[0].item()
    max_n = max(n_Rr, n_Rs)
    data['Rr'] = data['Rr'][:, :max_n, :]
    data['Rs'] = data['Rs'][:, :max_n, :]
    return data

def construct_relations(states, state_mask, tool_mask, adj_thresh_range=[0.1, 0.2], max_nR=500, adjacency=None, pushing_direction=None):
    # construct relations (density as hyperparameter)
    bsz = states.shape[0] # states: B, n_his, N, 3
    adj_thresh = np.random.uniform(*adj_thresh_range, (bsz,))
    
    Rr, Rs = construct_edges_from_states(states[:, -1], adj_thresh,
                                         mask=state_mask, tool_mask=tool_mask, no_self_edge=True,
                                         pushing_direction=pushing_direction)
    assert Rr[:, -1].sum() > 0
    Rr, Rs = Rr.detach(), Rs.detach()
    return Rr, Rs


def train(config):
    train_config = config['train_config']
    model_config = config['model_config']
    dataset_config = config['dataset_config']
    material_config = config['material_config']
    
    torch.autograd.set_detect_anomaly(True)
    set_seed(train_config['random_seed'])
    device = train_config['device']
    print(f"device: {device}")

    os.makedirs(train_config['out_dir'], exist_ok=True)
    os.makedirs(os.path.join(train_config['out_dir'], 'checkpoints'), exist_ok=True)

    # data loader
    phases = train_config['phases']
    dataset_config['n_his'] = train_config['n_his']
    dataset_config['n_future'] = train_config['n_future']
    datasets = {phase: DynDataset(
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

    # model
    model_config['n_his'] = train_config['n_his']
    model = DynamicsPredictor(model_config, material_config, device)
    model.to(device)

    # loss function and optimizer
    mse_loss = torch.nn.MSELoss()
    loss_funcs = [(mse_loss, 1)]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # load checkpoint if exists
    if train_config['load_model']:
        print(f'Loading model from {train_config["model_checkpoint"]} and {train_config["optimizer_checkpoint"]}')
        model_checkpoint = torch.load(train_config['model_checkpoint'])
        model.load_state_dict(model_checkpoint)
        optimizer_checkpoint = torch.load(train_config['optimizer_checkpoint'])
        optimizer.load_state_dict(optimizer_checkpoint)
    
    loss_plot_list_train, loss_plot_list_valid = [], []
    for epoch in range(train_config['n_epochs']):
        time1 = time.time()
        for phase in phases:
            with grad_manager(phase):
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                loss_sum_list = []
                n_iters = train_config['n_iters_per_epoch'][phase] \
                        if train_config['n_iters_per_epoch'][phase] != -1 else len(datasets[phase])
                for i in range(n_iters):
                    data = next(dataloaders[phase]) # graph
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                    
                    data = {key: data[key].to(device) for key in data.keys()}
                    loss_sum = 0
                    
                    future_state = data['state_future']  # (B, n_future, n_p, 3)
                    # future_mask = data['state_future_mask']  # (B, n_future)
                    future_tool = data['tool_future']  # (B, n_future-1, n_p+n_s, 3)
                    future_action = data['action_future']  # (B, n_future-1, n_p+n_s, 3)
                    
                    # print(f"future_state: {future_state.shape}, future_mask: {future_mask.shape}")
                    # print(f"future_tool: {future_tool.shape}, future_action: {future_action.shape}")
                    
                    state_mask = data['state_mask']
                    tool_mask = data['tool_mask']
                    
                    # print(f"state_mask: {state_mask.shape}, tool_mask: {tool_mask.shape}")
                    
                    for fi in range(train_config['n_future']):
                        gt_state = future_state[:, fi].clone() # (B, n_p, 3)
                        # gt_mask = future_mask[:, fi].clone() # (B,)
                        
                        # construct edges/relations
                        data['Rr'], data['Rs'] = construct_relations(data['state'], state_mask, tool_mask,
                                            adj_thresh_range=dataset_config['datasets'][0]['adj_radius_range'],
                                            pushing_direction=data['pushing_direction'])
                        # print(f"Rr: {data['Rr'].shape}, Rs: {data['Rs'].shape}")
                        # print(f"number of states: {data['state_future'].shape}")
                        
                        # predict state
                        pred_state, pred_motion = model(**data)
                        pred_state_p = pred_state[:, :gt_state.shape[1], :3].clone()
                        
                        loss = [weight * func(pred_state_p, gt_state) for func, weight in loss_funcs]
                        loss_sum += sum(loss)
                        
                        if fi < train_config['n_future'] - 1:
                            # build next graph
                            next_tool = future_tool[:, fi].clone() # (B, n_p+n_s, 3)
                            next_action = future_action[:, fi].clone() # (B, n_p+n_s, 3)
                            
                            next_state = next_tool.unsqueeze(1) # (B, 1, n_p+n_s, 3)
                            next_state[:, -1, :pred_state_p.shape[1]] = pred_state_p # (B, 1, n_p+n_s, 3)
                            next_state = torch.cat([data['state'][:, 1:], next_state], dim=1) # (B, n_his, n_p+n_s, 3)
                            data["state"] = next_state # (B, n_his, N+M, state_dim)
                            data["action"] = next_action
                    
                    if phase == 'train':
                        loss_sum.backward()
                        optimizer.step()
                        if i % train_config['log_interval'] == 0:
                            print(f'epoch {epoch}, iter {i}: loss = {loss_sum.item()}')
                            loss_sum_list.append(loss_sum.item())
                    if phase == 'valid':
                        loss_sum_list.append(loss_sum.item())
                
                if phase == 'valid':
                    print(f'\nEpoch {epoch}, valid loss {np.mean(loss_sum_list)}')
                
                if phase == 'train':
                    loss_plot_list_train.append(np.mean(loss_sum_list))
                if phase == 'valid':
                    loss_plot_list_valid.append(np.mean(loss_sum_list))
        
        if ((epoch + 1) < 100 and (epoch + 1) % 10 == 0) or (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(train_config['out_dir'], 'checkpoints', f'model_{(epoch + 1)}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(train_config['out_dir'], 'checkpoints', f'optim_{(epoch + 1)}.pth'))
        torch.save(model.state_dict(), os.path.join(train_config['out_dir'], 'checkpoints', f'latest.pth'))
        torch.save(optimizer.state_dict(), os.path.join(train_config['out_dir'], 'checkpoints', f'latest_optim.pth'))
        
        # plot figures
        plt.figure(figsize=(20, 5))
        plt.plot(loss_plot_list_train, label='train')
        plt.plot(loss_plot_list_valid, label='valid')
        # cut off figure
        ax = plt.gca()
        y_min = min(min(loss_plot_list_train), min(loss_plot_list_valid))
        y_min = min(loss_plot_list_valid)
        y_max = min(3 * y_min, max(max(loss_plot_list_train), max(loss_plot_list_valid)))
        ax.set_ylim([0, y_max])
        # save
        plt.legend()
        plt.savefig(os.path.join(train_config['out_dir'], 'loss.png'), dpi=300)
        plt.close()
        
        time2 = time.time()
        print(f'Epoch {epoch} time: {time2 - time1}\n')
                        
                        
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/debug.yaml')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    train(config)
