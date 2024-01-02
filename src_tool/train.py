import os
import time
import sys
import numpy as np

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.base_config import gen_args
from gnn.model import DynamicsPredictor
from gnn.utils import set_seed, umeyama_algorithm
from dataset import GranularToolDynDataset, construct_edges_from_states

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

def construct_relations(states, state_mask, eef_mask, adj_thresh_range=[0.1, 0.2], max_nR=500, adjacency=None):
    # construct relations (density as hyperparameter)
    bsz = states.shape[0] # states: B, n_his, N, 3
    adj_thresh = np.random.uniform(*adj_thresh_range, (bsz,))
    
    Rr, Rs = construct_edges_from_states(states[:, -1], adj_thresh * 1.5,
                                         mask=state_mask, eef_mask=eef_mask, no_self_edge=True)
    assert Rr[:, -1].sum() > 0
    Rr, Rs = Rr.detach(), Rs.detach()
    return Rr, Rs
    

def train(out_dir, data_dirs, prep_save_dir=None, material='carrots', ratios=None):
    torch.autograd.set_detect_anomaly(True)
    args = gen_args()
    set_seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f'device: {device}')
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    phases = ['train', 'valid']
    
    # training hyperparameters
    batch_size = 64
    n_epoch = 10
    n_iters_per_epoch = {'train': 250, 'valid': 25}
    log_interval = 50
    
    # data preprocessing hyperparameters
    n_his = 4
    n_future = 3
    dist_thresh = 0.2
    
    adj_thresh_range = [args.adj_thresh_min, args.adj_thresh_max]
    print(f"adj_thresh_range: {adj_thresh_range}")
    
    data_kwargs = {"train": {
            "n_his": n_his,
            "n_future": n_future,
            "dist_thresh": dist_thresh,
            "adj_thresh_range": adj_thresh_range, # adj_thresh_range, # construct edge
            "fps_radius_range": adj_thresh_range, # fps sampling: determine number of nodes
            "max_n": 1, # number of objects
            "max_nobj": 100, # number of particles per object
            "max_ntool": 100, # number of eef particles
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
            "max_ntool": 100,
            "max_nR": 500,
            "ratios": [0.9, 1],
            "phys_noise": 0.0,
        }
    }
    
    datasets = {phase: GranularToolDynDataset(data_dirs, prep_save_dir, phase, **data_kwargs[phase]) for phase in phases}
    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=1,
    ) for phase in phases}
    dataloaders = {phase: dataloader_wrapper(dataloaders[phase], phase) for phase in phases}
    
    ## set args TODO: understand the meaning of these args
    # particle encoder
    args.attr_dim = 2
    args.n_his = n_his
    args.state_dim = 0 # abd (x, y, z) # not used in particle encoder and thus set to 0
    args.offset_dim = 0 # similar with the state_dim
    args.action_dim = 3
    args.pstep = 3
    args.time_step = 1
    args.dt = 1. / 60.
    args.sequence_length = 4
    args.phys_dim = 2
    args.density_dim = 0
    
    # relation encoder
    args.rel_particle_dim = 0
    args.rel_attr_dim = 2
    args.rel_group_dim = 1 # sum of difference of group one-hot vector
    args.rel_distance_dim = 3 # no distance
    args.rel_density_dim = 0 # no density
    
    # TODO rel canonical (not used)
    args.rel_canonical_distance_dim = 0
    args.rel_canonical_attr_dim = 0
    args.rel_canonical_thresh = 3 * data_kwargs['train']['adj_thresh_range'][0]
    
    # TODO rel can attr
    args.rel_can_attr_dim = 0 
    
    # TODO physencoder
    args.use_vae = False
    args.phys_encode = False
    
    ## model
    model_kwargs = {}
    model_kwargs.update({
        "predict_rigid": False,
        "predict_non_rigid": True,
        "rigid_out_dim": 0,
        "non_rigid_out_dim": 3,
    })
    model = DynamicsPredictor(args, verbose=False, **model_kwargs)
    model.to(device)
    
    mse_loss = torch.nn.MSELoss()
    loss_funcs = [(mse_loss, 1)]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_plot_list_train, loss_plot_list_valid = [], []
    for epoch in range(n_epoch):
        time1 = time.time()
        for phase in phases:
            with grad_manager(phase):
                if phase == 'train':
                    model.train()
                    if args.phys_encode:
                        phys_encoded_noise = 0.01
                    else:
                        phys_encoded_noise = 0.0
                else:
                    model.eval()
                    phys_encoded_noise = 0.0
                loss_sum_list = []
                n_iters = n_iters_per_epoch[phase] if n_iters_per_epoch[phase] != -1 else len(datasets[phase])
                for i in range(n_iters):
                    data = next(dataloaders[phase]) # graph
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                    
                    data = {key: data[key].to(device) for key in data.keys()}
                    loss_sum = 0
                    
                    data['phys_encoded_noise'] = phys_encoded_noise
                    
                    future_state = data['state_future']  # (B, n_future, n_p, 3)
                    future_mask = data['state_future_mask']  # (B, n_future)
                    future_eef = data['eef_future']  # (B, n_future-1, n_p+n_s, 3)
                    future_action = data['action_future']  # (B, n_future-1, n_p+n_s, 3)
                    
                    state_mask = data['state_mask']
                    eef_mask = data['eef_mask']
                    
                    for fi in range(n_future):
                        gt_state = future_state[:, fi].clone() # (B, n_p, 3)
                        gt_mask = future_mask[:, fi].clone() # (B,)
                        
                        # construct edges/relations
                        data['Rr'], data['Rs'] = construct_relations(data['state'], state_mask, eef_mask,
                                                                     adj_thresh_range=data_kwargs[phase]['adj_thresh_range'],)
                        # print(f"Rr: {data['Rr'].shape}, Rs: {data['Rs'].shape}")
                        # print(f"number of states: {data['state_future'].shape}")
                        
                        # predict state
                        pred_state, pred_motion = model(**data)
                        pred_state_p = pred_state[:, :gt_state.shape[1], :3].clone()
                        
                        loss = [weight * func(pred_state_p[gt_mask], gt_state[gt_mask]) for func, weight in loss_funcs]
                        loss_sum += sum(loss)
                        
                        if fi < n_future - 1:
                            # build next graph
                            next_eef = future_eef[:, fi].clone() # (B, n_p+n_s, 3)
                            next_action = future_action[:, fi].clone() # (B, n_p+n_s, 3)
                            next_state = next_eef.unsqueeze(1) # (B, 1, n_p+n_s, 3)
                            next_state[:, -1, :pred_state_p.shape[1]] = pred_state_p # (B, 1, n_p+n_s, 3)
                            next_state = torch.cat([data['state'][:, 1:], next_state], dim=1) # (B, n_his, n_p+n_s, 3)
                            data["state"] = next_state # (B, n_his, N+M, state_dim)
                            data["action"] = next_action
                    
                    if phase == 'train':
                        loss_sum.backward()
                        optimizer.step()
                        if i % log_interval == 0:
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
            torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoints', f'model_{(epoch + 1)}.pth'))
        torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoints', f'latest.pth'))
        torch.save(optimizer.state_dict(), os.path.join(out_dir, 'checkpoints', f'latest_optim.pth'))
        
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
        plt.savefig(os.path.join(out_dir, 'loss.png'), dpi=300)
        plt.close()
        
        time2 = time.time()
        print(f'Epoch {epoch} time: {time2 - time1}\n')
                        
                        
if __name__ == '__main__':
    args = gen_args()
    
    train(args.out_dir, args.data_dir, prep_save_dir=args.prep_save_dir)