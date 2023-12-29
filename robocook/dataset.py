import os
import glob
import numpy as np

import torch

from data_utils import *

data_names = ["positions", "shape_quats", "scene_params"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

n_particles = 300
floor_dim = 9

time_step = 1
data_time_step = 1
n_his = 4
sequence_length = 4

tool_type = 'gripper_sym_rod_robot_v4_surf_nocorr_full_normal_keyframe=16'
dy_data_path = f'/mnt/sda/robocook/dynamics/dynamics_data/data_{tool_type}'

phase = 'train'
data_dir = os.path.join(dy_data_path, phase)
stat_path = os.path.join(dy_data_path, "..", "stats.h5")
vid_path_list = sorted(glob.glob(os.path.join(data_dir, "*")))
# print(vid_path_list)

state_data_list, action_data_list = [], []
n_frames_min = float('inf')
dataset_len = 0

for vid_path_idx in range(1):
    vid_path = vid_path_list[vid_path_idx]
    gt_vid_path = vid_path
    print(f"gt_vid_path: {gt_vid_path}")
    
    frame_start = 0
    n_frames = len(glob.glob(os.path.join(gt_vid_path, "*.h5"))) # 16
    # print(f"n_frames: {n_frames}")
    n_frames_min = min(n_frames_min, n_frames)
    
    gt_state_list, gt_action_list = [], []
    for i in range(n_frames):
        gt_frame_data = load_data(data_names, os.path.join(gt_vid_path, f"{str(i).zfill(3)}.h5"))[0] # positions 
        # print(f"gt_frame_data.shape: {gt_frame_data.shape}") # (493, 6)
        gt_state = torch.tensor(gt_frame_data, device=device, dtype=torch.float32)
        # print(f"gt_state.shape: {gt_state.shape}") # (493, 6)
        gt_state_list.append(gt_state)
        gt_action_list.append(gt_state[n_particles+floor_dim :])
    
    action_data_list.append(torch.stack(gt_action_list))
    # print(action_data_list[0].shape) # (16, 184, 6)
    
    state_seq_list = []
    # print(n_frames - time_step * data_time_step * (n_his + sequence_length - 1)) # 9
    for i in range(frame_start, n_frames - time_step * data_time_step * (n_his + sequence_length - 1)):
        state_seq = []
        # history frames
        for j in range(i, i + time_step * (n_his - 1) + 1, time_step):
            state_seq.append(gt_state_list[j])
        
        # frames to predict
        for j in range(
            i + time_step * n_his,
            i + time_step * (n_his + sequence_length - 1) + 1, 
            time_step
        ):
            state_seq.append(gt_state_list[j])
        
        dataset_len += 1
        state_seq_list.append(torch.stack(state_seq))
    
    state_data_list.append(torch.stack(state_seq_list))
    print(f"{phase} -> number of sequences: {dataset_len}")
        



