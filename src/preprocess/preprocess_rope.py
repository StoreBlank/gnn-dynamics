import os
import glob
import json
import numpy as np
import open3d as o3d
import torch

from dgl.geometry import farthest_point_sampler
from utils import quaternion_to_rotation_matrix, fps_rad_idx, pad

def extract_kp(data_dir, episode_idx, start_frame, end_frame):
    # obtain object keypoints
    obj_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    obj_ptcl_start, obj_ptcl_end = obj_ptcl[start_frame], obj_ptcl[end_frame]
    obj_kp = np.stack([obj_ptcl_start, obj_ptcl_end], axis=0)
    
    # obtain tool keypoints
    tool_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/processed_eef_states.npy"))
    tool_ptcl_start, tool_ptcl_end = tool_ptcl[start_frame], tool_ptcl[end_frame]
    tool_kp = np.stack([tool_ptcl_start, tool_ptcl_end], axis=0)
    
    return obj_kp, tool_kp

def extract_kp_single_frame(data_dir, episode_idx, frame_idx):
    # obtain object keypoints
    obj_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    # print(f"epi_idx {episode_idx}, frame_idx {frame_idx}, obj_ptcls: {obj_ptcls.shape}")
    obj_ptcl = obj_ptcls[frame_idx]
    obj_kp = np.array([obj_ptcl])
    
    # obtain tool keypoints
    tool_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/processed_eef_states.npy"))
    tool_kp = tool_ptcl[frame_idx]
    tool_kp = np.array(tool_kp)
    
    return obj_kp, tool_kp

"""
Preprocess data to save the following:
    - frame_pairs: a directory containing the start-end frame pairs for each push.
        - {epi_idx}_{push_idx}.txt: save the push pairs for each frame: (n_his - 1, curr, n_future)
    - phys_range.txt: a file containing the min and max of the physical parameters.
    - metadata.txt: a file containing the metadata of the preprocessed data.
"""

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
        # load particle
        particles_pos = np.load(os.path.join(data_dir, f"episode_{epi_idx}/particles_pos.npy"))
        num_frames = particles_pos.shape[0]
        # num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{epi_idx}/camera_0/*_color.jpg"))))
        print(f"Processing episode {epi_idx}, num_frames: {num_frames}")
        
        # load info
        steps = np.load(os.path.join(data_dir, f"episode_{epi_idx}/steps.npy"))
        
        eef_states = np.load(os.path.join(data_dir, f"episode_{epi_idx}/eef_states.npy"))
        assert eef_states.shape[0] == num_frames
        eef_pos = np.zeros((num_frames, 3))
        for i in range(num_frames):
            eef_pos_0 = eef_states[i, 0:3]
            eef_quat = eef_states[i, 6:10]
            eef_rot = quaternion_to_rotation_matrix(eef_quat)
            eef_pos[i] = eef_pos_0 + np.dot(eef_rot, np.array([0, 0, 1.]))
        
        physics_path = os.path.join(data_dir, f"episode_{epi_idx}/property_params.json")
        with open(physics_path, "r") as f:
            properties = json.load(f)
        phys_param = np.array([
            # properties['particle_radius'],
            # properties['num_particles'],
            properties['length'],
            properties['thickness'],
            properties['dynamic_friction'],
            properties['cluster_spacing']
        ]).astype(np.float32)
        phys_params.append(phys_param)
        
        # get start-end pairs
        frame_idxs = []
        cnts = [0]
        cnt = 0
        for fj in range(num_frames):
            curr_step = None
            for si in range(len(steps) - 1):
                if fj >= steps[si] and fj <= steps[si + 1] - 2:
                    curr_step = si
                    break
            else:
                continue # this frame is not valid
            assert curr_step is not None
        
            curr_frame = fj
            start_frame = steps[curr_step]
            end_frame = steps[curr_step + 1] - 2
            
            # search backward (n_his)
            eef_particles_curr = eef_pos[curr_frame]
            frame_traj = [curr_frame]
            fi = fj
            while fi >= start_frame:
                eef_particles_fi = eef_pos[fi]
                x_curr, z_curr = eef_particles_curr[0], eef_particles_curr[2]
                x_fi, z_fi = eef_particles_fi[0], eef_particles_fi[2]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)
                if dist_curr >= dist_thresh:
                    frame_traj.append(fi)
                    eef_particles_curr = eef_particles_fi
                fi -= 1
                if len(frame_traj) == n_his:
                    break
            else: 
                # pad to n_his
                frame_traj = frame_traj + [frame_traj[-1]] * (n_his - len(frame_traj))
            frame_traj = frame_traj[::-1]
            
            # search forward (n_future)
            eef_particles_curr = eef_pos[curr_frame]
            fi = fj
            while fi <= end_frame:
                eef_particles_fi = eef_pos[fi]
                x_curr, z_curr = eef_particles_curr[0], eef_particles_curr[2]
                x_fi, z_fi = eef_particles_fi[0], eef_particles_fi[2]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)
                if dist_curr >= dist_thresh or (fi == end_frame and dist_curr >= 0.75 * dist_thresh):
                    frame_traj.append(fi)
                    eef_particles_curr = eef_particles_fi
                fi += 1
                if len(frame_traj) == n_his + n_future:
                    cnt += 1
                    break
            else:
                # When assuming quasi-static, we can pad to n_his + n_future
                frame_traj = frame_traj + [frame_traj[-1]] * (n_his + n_future - len(frame_traj))
                cnt += 1
            
            frame_idxs.append(frame_traj)

            # push_centered
            if fj == end_frame:
                cnts.append(cnt)
                frame_idxs = np.array(frame_idxs)
                np.savetxt(os.path.join(frame_idx_dir, f"{epi_idx}_{curr_step}.txt"), frame_idxs, fmt="%d")
                print(f"episode {epi_idx}, push {curr_step} has {cnts[curr_step+1]-cnts[curr_step]} pushes.")
                frame_idxs = []
    
    # save phys_params stat
    phys_params = np.stack(phys_params, axis=0)
    phys_params_max = np.max(phys_params, axis=0)
    phys_params_min = np.min(phys_params, axis=0)
    phys_params_range = np.stack([phys_params_min, phys_params_max], axis=0)
    print(f"phys_params_range: {phys_params_range}")
    np.savetxt(os.path.join(save_dir, "phys_range.txt"), phys_params_range)
    
def extract_eef_points(data_dir):
    # calculate the number of episodes folder in the data directory
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Preprocessing starts. Number of episodes: {num_episodes}")
    
    eef_point_pos = np.array([[0., 0., 1.]])
    n_eef_points = eef_point_pos.shape[0]
    
    for epi_idx in range(num_episodes):
        # load particle
        particles_pos = np.load(os.path.join(data_dir, f"episode_{epi_idx}/particles_pos.npy"))
        num_frames = particles_pos.shape[0]
        # num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{epi_idx}/camera_0/*_color.jpg"))))
        print(f"Processing episode {epi_idx}, num_frames: {num_frames}")
        
        # load the eef states
        eef_states = np.load(os.path.join(data_dir, f"episode_{epi_idx}/eef_states.npy"))
        assert eef_states.shape[0] == num_frames
        
        # extract eef points
        processed_eef_states = np.zeros((num_frames, n_eef_points, 3))
        for frame_idx in range(num_frames):
            eef_state = eef_states[frame_idx]
            eef_pos_0 = eef_state[0:3]
            eef_quat = eef_state[6:10]
            eef_rot = quaternion_to_rotation_matrix(eef_quat)
            for j in range(n_eef_points):
                eef_pos = eef_pos_0 + np.dot(eef_rot, eef_point_pos[j])
                processed_eef_states[frame_idx, j] = eef_pos
            
        # save the processed eef states
        np.save(os.path.join(data_dir, f"episode_{epi_idx}/processed_eef_states.npy"), processed_eef_states)
        
if __name__ == "__main__":
    data_name = "rope"
    data_dir_list = [
        # f"/mnt/nvme1n1p1/baoyu/data/{data_name}"
        f"/mnt/sda/data/{data_name}"
    ]
    save_dir_list = [
        # f"/mnt/nvme1n1p1/baoyu/preprocess/{data_name}"
        f"/mnt/sda/preprocess/{data_name}"
    ]
    dist_thresh = 0.25 #2.5cm
    n_his = 4
    n_future = 3
    
    for data_dir, save_dir in zip(data_dir_list, save_dir_list):
        if os.path.isdir(data_dir):
            os.makedirs(save_dir, exist_ok=True)
            print("================extract_pushes================")
            extract_pushes(data_dir, save_dir, dist_thresh, n_his, n_future)
            print("==============================================")
            print("================extract eef================")
            extract_eef_points(data_dir)
            print("==============================================")
        # save metadata
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
            f.write(f'{dist_thresh},{n_future},{n_his}')