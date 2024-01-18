import os
import glob
import json
import numpy as np
import open3d as o3d
import torch

from dgl.geometry import farthest_point_sampler
from utils import quaternion_to_rotation_matrix, fps_rad_idx, pad

def extract_physics_params(data_dir):
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Preprocessing box points starts. Number of episodes: {num_episodes}")

    for epi_idx in range(num_episodes):
        box_com = np.load(os.path.join(data_dir, f"episode_{epi_idx:03d}/box_com.npy")) # box_size, box_com
        com = box_com[1]
        property_params = {
            'com_x': com[0],
            'com_y': com[1],
        }
        print(f"Processing episode {epi_idx:03d}, physics_params: {property_params}")
        with open(os.path.join(data_dir, f"episode_{epi_idx:03d}/property_params.json"), 'w') as f:
            json.dump(property_params, f)

def extract_box_points(data_dir):
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Preprocessing box points starts. Number of episodes: {num_episodes}")
    
    for epi_idx in range(num_episodes): 
        box_com = np.load(os.path.join(data_dir, f"episode_{epi_idx:03d}/box_com.npy"))
        box_size = box_com[0]
        
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{epi_idx:03d}/images/*.png"))))
        box_states = np.load(os.path.join(data_dir, f"episode_{epi_idx:03d}/box_states.npy"))
        assert box_states.shape[0] == num_frames
        print(f"Processing episode {epi_idx:03d}, num_frames: {num_frames}")
        
        # extract points from the four corners of the box
        box_points_in_box = np.array([
            [-box_size[0] / 2, box_size[1] / 2],
            [box_size[0] / 2, box_size[1] / 2],
            [box_size[0] / 2, -box_size[1] / 2],
            [-box_size[0] / 2, -box_size[1] / 2],
        ])
        
        num_box_points = box_points_in_box.shape[0]
        processed_box_pos = np.zeros((num_frames, num_box_points, 2))
        for i in range(num_frames):
            box_state = box_states[i]
            box_pos = box_state[0:2]
            box_rad = box_state[2]
            
            for j in range(num_box_points):
                point = box_points_in_box[j]
                point_rotated_x = point[0] * np.cos(box_rad) + point[1] * np.sin(box_rad)
                point_rotated_y = -point[0] * np.sin(box_rad) + point[1] * np.cos(box_rad)
                point_rotated = np.array([point_rotated_x, point_rotated_y])
                point_world = box_pos + point_rotated
                processed_box_pos[i, j] = point_world
        
        np.save(os.path.join(data_dir, f"episode_{epi_idx:03d}/processed_box_pos.npy"), processed_box_pos)
        print(f"episode {epi_idx:03d} processed_box_pos saved.")

def extract_pushes(data_dir, save_dir, dist_thresh, n_his, n_future):
    """
    Args:
        data_dir (str): directory of the raw data
        save_dir (str): directory to save the processed data
        dist_thresh (float): distance threshold to determine a push pair
        n_his (int): number of frames to look back
        n_future (int): number of frames to look forward
    """
    
    frame_idx_dir = os.path.join(save_dir, "frame_pairs")
    os.makedirs(frame_idx_dir, exist_ok=True)
    
    # calculate the number of episodes folder in the data directory
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Preprocessing starts. Number of episodes: {num_episodes}")
    
    phys_params = []
    
    for epi_idx in range(num_episodes):
        # load states
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{epi_idx:03d}/images/*.png"))))
        box_states = np.load(os.path.join(data_dir, f"episode_{epi_idx:03d}/box_states.npy")) # (num_frames, 3): (x, y, theta)
        eef_states = np.load(os.path.join(data_dir, f"episode_{epi_idx:03d}/eef_states.npy")) # (num_frames, 2): (x, y)
        assert num_frames == box_states.shape[0] == eef_states.shape[0]
        print(f"Processing episode {epi_idx:03d}, num_frames: {num_frames}")
        
        # load physics parameters: center of mass (com)
        box_com = np.load(os.path.join(data_dir, f"episode_{epi_idx:03d}/box_com.npy")) # box_size, box_com
        com = box_com[1]
        phys_param = np.array([com[0], com[1]]).astype(np.float32)
        phys_params.append(phys_param)
        
        # get start-end pairs
        frame_idxs = []
        cnt = 0
        for fj in range(num_frames):
            curr_frame = fj
            # 1 step per episode
            start_frame = 0
            end_frame = num_frames - 1
            curr_step = 0
            
            # search backward (n_his)
            eef_curr = eef_states[curr_frame]
            frame_traj = [curr_frame]
            fi = fj
            while fi >= start_frame:
                eef_fi = eef_states[fi]
                dist_curr = np.linalg.norm(eef_curr - eef_fi)
                if dist_curr >= dist_thresh:
                    frame_traj.append(fi)
                    eef_curr = eef_fi
                fi -= 1
                if len(frame_traj) == n_his:
                    break
            else:
                # pad to n_his
                frame_traj = frame_traj + [frame_traj[-1]] * (n_his - len(frame_traj))
            frame_traj = frame_traj[::-1]
        
            # search forward (n_future)
            eef_curr = eef_states[curr_frame]
            fi = fj
            while fi <= end_frame:
                eef_fi = eef_states[fi]
                dist_curr = np.linalg.norm(eef_curr - eef_fi)
                if dist_curr >= dist_thresh or (fi == end_frame and dist_curr >= 0.75 * dist_thresh):
                    frame_traj.append(fi)
                    eef_curr = eef_fi
                fi += 1
                if len(frame_traj) == n_his + n_future:
                    cnt += 1
                    break
            else:
                # When assuming quasi-static, we can pad to n_his + n_future
                frame_traj = frame_traj + [frame_traj[-1]] * (n_his + n_future - len(frame_traj))
                cnt += 1
            
            frame_idxs.append(frame_traj)
            
            # push centered
            if fj == end_frame:
                frame_idxs = np.array(frame_idxs)
                np.savetxt(os.path.join(frame_idx_dir, f"{epi_idx:03d}_{curr_step}.txt"), frame_idxs, fmt="%d")
                print(f"episode {epi_idx:03d}, push {curr_step} has {cnt} pushes.")
                frame_idxs = []
    
    # save phys_params stat
    phys_params = np.stack(phys_params, axis=0)
    phys_params_max = np.max(phys_params, axis=0)
    phys_params_min = np.min(phys_params, axis=0)
    phys_params_range = np.stack([phys_params_min, phys_params_max], axis=0)
    print(f"phys_params_range: {phys_params_range}")
    np.savetxt(os.path.join(save_dir, "phys_range.txt"), phys_params_range)
    

if __name__ == "__main__":
    data_name = "box"
    data_dir_list = [
        f"/mnt/nvme1n1p1/baoyu/data/{data_name}"
    ]
    save_dir_list = [
        f"/mnt/nvme1n1p1/baoyu/preprocess_010/{data_name}"
    ]
    dist_thresh = 10 #10mm=1cm
    n_his = 4
    n_future = 3
    
    for data_dir, save_dir in zip(data_dir_list, save_dir_list):
        if os.path.isdir(data_dir):
            os.makedirs(save_dir, exist_ok=True)
            # print("================extract_pushes================")
            # extract_pushes(data_dir, save_dir, dist_thresh, n_his, n_future)
            # print("==============================================")
            # print("================extract_box_points================")
            # extract_box_points(data_dir)
            # print("==================================================")
            print("================extract_physics_params================")
            extract_physics_params(data_dir)
            print("======================================================")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
            f.write(f'{dist_thresh},{n_future},{n_his}')