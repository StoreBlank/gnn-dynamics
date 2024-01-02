import os
import numpy as np

def extract_kp(data_dir, episode_idx, start_frame, end_frame):
    # obtain object keypoints
    obj_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    obj_ptcl_start, obj_ptcl_end = obj_ptcl[start_frame], obj_ptcl[end_frame]
    obj_kp = np.stack([obj_ptcl_start, obj_ptcl_end], axis=0)
    
    # obtain static tool keypoints
    static_tool_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/dustpan_points.npy"))
    static_tool_kp = np.stack([static_tool_ptcl[start_frame], static_tool_ptcl[end_frame]], axis=0)
    
    # obtain dynamic tool keypoints
    dynamic_tool_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/sponge_points.npy"))
    dynamic_tool_kp = np.stack([dynamic_tool_ptcl[start_frame], dynamic_tool_ptcl[end_frame]], axis=0)
    
    return obj_kp, static_tool_kp, dynamic_tool_kp

def extract_kp_single_frame(data_dir, episode_idx, frame_idx):
    # obtain object keypoints
    obj_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    obj_ptcl = obj_ptcls[frame_idx]
    obj_kp = np.array([obj_ptcl])
    
    # obtain static tool keypoints
    static_tool_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/dustpan_points.npy"))
    static_tool_ptcl = static_tool_ptcls[frame_idx]
    static_tool_kp = np.array([static_tool_ptcl])
    
    # obtain dynamic tool keypoints
    dynamic_tool_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/sponge_points.npy"))
    dynamic_tool_ptcl = dynamic_tool_ptcls[frame_idx]
    dynamic_tool_kp = np.array([dynamic_tool_ptcl])
    
    return obj_kp, static_tool_kp, dynamic_tool_kp

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