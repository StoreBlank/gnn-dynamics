import os
import numpy as np

def extract_kp(data_dir, episode_idx, start_frame, end_frame):
    # obtain object keypoints
    obj_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    obj_ptcl_start, obj_ptcl_end = obj_ptcl[start_frame], obj_ptcl[end_frame]
    obj_kp = np.stack([obj_ptcl_start, obj_ptcl_end], axis=0)
    
    # obatin end-effector keypoints
    # y = 0.5 #TODO: check whether it is correct
    eef_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/eef_pos.npy"))
    eef_ptcl_start, eef_ptcl_end = eef_ptcl[start_frame], eef_ptcl[end_frame]
    x_start, z_start, y_start = eef_ptcl_start[0], eef_ptcl_start[1], eef_ptcl_start[2]
    x_end, z_end, y_end = eef_ptcl_end[0], eef_ptcl_end[1], eef_ptcl_end[2]
    
    pos_start = np.array([[x_start, y_start, z_start]])
    pos_end = np.array([[x_end, y_end, z_end]])
    eef_kp = np.stack([pos_start, pos_end], axis=0)
    eef_kp[:, :, 2] *= -1
    return obj_kp, eef_kp

def extract_kp_single_frame(data_dir, episode_idx, frame_idx):
    # obtain object keypoints
    obj_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    obj_ptcl = obj_ptcls[frame_idx]
    obj_kp = [obj_ptcl]
    
    # obatin end-effector keypoints
    # y = obj_kp[0][:, 1].mean() #TODO: check whether it is correct
    eef_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/eef_pos.npy"))
    eef_ptcl = eef_ptcls[frame_idx]
    x, z, y = eef_ptcl[0], eef_ptcl[1], eef_ptcl[2]
    eef_kp = np.array([[x, y, -z]])
    return obj_kp, eef_kp

def fps_rad_idx(pcd, radius):
    # pcd: (n, 3) numpy array
    # pcd_fps: (-1, 3) numpy array
    # radius: float
    rand_idx = np.random.randint(pcd.shape[0])
    pcd_fps_lst = [pcd[rand_idx]]
    idx_lst = [rand_idx]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while dist.max() > radius:
        pcd_fps_lst.append(pcd[dist.argmax()])
        idx_lst.append(dist.argmax())
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    idx_lst = np.stack(idx_lst, axis=0)
    return pcd_fps, idx_lst

def pad(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = np.zeros((max_dim, x.shape[1]), dtype=np.float32)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = np.zeros((x.shape[0], max_dim, x.shape[2]), dtype=np.float32)
        x_pad[:, :x_dim] = x
    return x_pad