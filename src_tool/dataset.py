import os
import glob
import json
import numpy as np

import torch 
from torch.utils.data import Dataset

from dgl.geometry import farthest_point_sampler

def load_pairs(pairs_path, episode_range):
    pair_lists = []
    for episode_idx in episode_range:
        n_pushes = len(list(glob.glob(os.path.join(pairs_path, f'{episode_idx}_*.txt'))))
        for push_idx in range(n_pushes):
            frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}_{push_idx}.txt'))
            if len(frame_pairs.shape) == 1: continue
            episodes = np.ones((frame_pairs.shape[0], 1)) * episode_idx
            pairs = np.concatenate([episodes, frame_pairs], axis=1)
            pair_lists.extend(pairs)
    pair_lists = np.array(pair_lists).astype(int)
    return pair_lists

class GranularToolDynDataset(Dataset):
    def __init__(
        self,
        data_dir, 
        prep_save_dir, 
        phase, 
        ratios, 
        dist_thresh,
        fps_radius_range, # radius for fps sampling
        adj_thresh_range, # threshold for constructing edges (not used here)
        n_future,
        n_his,
        max_n, # max number of objects
        max_nobj, # max number of object points
        max_ntool, # max number of points per tool
        max_nR, # max number of relations (not used here)
        phys_noise = 0.0,
        canonical = False,
    ):
        self.phase = phase
        self.data_dir = data_dir
        print(f'Setting up GranularToolDynDataset')
        print(f'Setting up {phase} dataset, data_dir: {len(data_dir)}')
        
        self.n_his = n_his
        self.n_future = n_future
        self.dist_thresh = dist_thresh
        
        self.max_n = max_n
        self.max_nobj = max_nobj
        self.max_ntool = max_ntool
        self.fps_radius_range = fps_radius_range # for object points
        self.phys_noise = phys_noise
        
        self.obj_kypts_paths = []
        self.eef_kypts_paths = []
        self.physics_paths = []
        
        # load kypts paths
        num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
        print(f"Found num_episodes: {num_episodes}")
        frame_count = 0
        for episode_idx in range(num_episodes):
            n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
            obj_kypts_path = os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy")
            physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property.json")
            
            self.obj_kypts_paths.append(obj_kypts_path)
            self.physics_paths.append(physics_path)
            
            frame_count += n_frames
        
        print(f'Found {frame_count} frames in {data_dir}')



