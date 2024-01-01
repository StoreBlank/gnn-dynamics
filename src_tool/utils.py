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

def quaternion_to_rotation_matrix(q):
    # Extract the values from q
    q1, q2, q3, w = q
    
    # First row of the rotation matrix
    r00 = 1 - 2 * (q2 ** 2 + q3 ** 2)
    r01 = 2 * (q1 * q2 - q3 * w)
    r02 = 2 * (q1 * q3 + q2 * w)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q3 * w)
    r11 = 1 - 2 * (q1 ** 2 + q3 ** 2)
    r12 = 2 * (q2 * q3 - q1 * w)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q2 * w)
    r21 = 2 * (q2 * q3 + q1 * w)
    r22 = 1 - 2 * (q1 ** 2 + q2 ** 2)
    
    # Combine all rows into a single matrix
    rotation_matrix = np.array([[r00, r01, r02],
                                [r10, r11, r12],
                                [r20, r21, r22]])
    
    return rotation_matrix