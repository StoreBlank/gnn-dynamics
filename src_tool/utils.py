import os
import numpy as np

def extract_kp(data_dir, episode_idx, start_frame, end_frame):
    # obtain object keypoints
    obj_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    obj_ptcl_start, obj_ptcl_end = obj_ptcl[start_frame], obj_ptcl[end_frame]
    obj_kp = np.stack([obj_ptcl_start, obj_ptcl_end], axis=0)
    
    # obatin end-effector keypoints
    eef_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/eef_pos.npy"))
    eef_ptcl_start, eef_ptcl_end = eef_ptcl[start_frame], eef_ptcl[end_frame]
    eef_kp = np.stack([eef_ptcl_start, eef_ptcl_end], axis=0)
    
    return obj_kp, eef_kp