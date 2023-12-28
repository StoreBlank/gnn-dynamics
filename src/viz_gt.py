import os
import glob
import cv2
import numpy as np
import torch
from dgl.geometry import farthest_point_sampler

from config.base_config import gen_args
from gnn.utils import set_seed
from dataset.dataset_carrots import pad
from dataset.utils import rgb_colormap, extract_kp_single_frame, fps_rad_idx

def viz_gt():
    """
    visualize ground truth keypoints (object and eef)
    """
    args = gen_args()
    set_seed(args.random_seed)
    
    # data hyperparameters
    n_future = 4
    n_his = 3
    
    fps_radius = 0.1
    max_n = 1
    max_nobj = 300
    max_neef = 1
    
    # visualization hyperparameters
    point_size = 3
    
    # others
    data_name = args.data_name
    data_dir = f"/mnt/sda/data/{data_name}"
    prep_save_dir = f"/mnt/sda/preprocess/{data_name}" 
    
    episode_idx = 24
    push_idx = 0
    start_idx = 0
    rollout_steps = 10
    
    colormap = rgb_colormap(repeat=100)
    
    save_dir = f"/mnt/sda/viz_gt/{data_name}-{episode_idx}_{push_idx}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Visualizing episode {episode_idx}, push {push_idx}.")
    print(f"Saving to {save_dir}")

    # load paris 
    pairs = np.loadtxt(os.path.join(prep_save_dir, 'frame_pairs', f'{episode_idx}_{push_idx}.txt')).astype(int)
    print(f'Found {len(pairs)} frame pairs for episode {episode_idx} push {push_idx}')
    
    # load keypoints
    n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
    obj_kypts_paths = os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy")
    eef_kypts_paths = os.path.join(data_dir, f"episode_{episode_idx}/eef_pos.npy")
    
    # load camera (only for visualization)
    intr_list = [None] * 4
    extr_list = [None] * 4
    for cam in range(4):
        os.makedirs(os.path.join(save_dir, f'camera_{cam}'), exist_ok=True)
        intr_list[cam] = np.load(os.path.join(data_dir, f"camera_intrinsic_params.npy"))[cam]
        extr_list[cam] = np.load(os.path.join(data_dir, f"camera_extrinsic_matrix.npy"))[cam]
    
    # get starting pair
    start_pair_idx = start_idx
    pair = pairs[start_pair_idx]
    print('starting pair', pair[:n_his-1])
    start = pair[n_his-1]
    end = pair[n_his]
    
    # get history keypoints
    obj_kps, eef_kps = [], []
    for i in range(n_his+1):
        frame_idx = pair[i]
        obj_kp, eef_kp = extract_kp_single_frame(data_dir, episode_idx, frame_idx)
        print(f"Frame {frame_idx} eef pos: {eef_kp}") # init true eef pos
        obj_kps.append(obj_kp)
        eef_kps.append(eef_kp)
    
    obj_kp_start = obj_kps[n_his-1]
    instance_num = len(obj_kp_start)
    assert instance_num == 1, 'only support single object'
    
    fps_idx_list = []
    for j in range(len(obj_kp_start)):
        # fps
        particle_tensor = torch.from_numpy(obj_kp_start[j]).float()[None, ...]
        fps_idx_tensor = farthest_point_sampler(particle_tensor, max_nobj, start_idx = np.random.randint(0, obj_kp_start[j].shape[0]))[0]
        fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32) # (max_nobj,)
        print(f"fps_idx_1: {fps_idx_1.shape}")
        
        # downsample to uniform radius
        downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
        _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius) # (max_nobj,)
        print(f"fps_idx_2: {fps_idx_2.shape}")
        fps_idx = fps_idx_1[fps_idx_2]
        fps_idx_list.append(fps_idx)
        
    # downsample to get current obj kp
    obj_kp_start = [obj_kp_start[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
    obj_kp_start = np.concatenate(obj_kp_start, axis=0) # (N, 3)
    obj_kp_num = obj_kp_start.shape[0]
    
    # load history states
    state_history = np.zeros((n_his, max_nobj + max_neef, obj_kp_start.shape[-1]), dtype=np.float32)
    for fi in range(n_his):
        obj_kp_his = obj_kps[fi]
        obj_kp_his = [obj_kp_his[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
        obj_kp_his = np.concatenate(obj_kp_his, axis=0)
        obj_kp_his = pad(obj_kp_his, max_nobj)
        state_history[fi, :max_nobj] = obj_kp_his
        
        eef_kp_his = eef_kps[fi]
        eef_kp_his = pad(eef_kp_his, max_neef)
        state_history[fi, max_nobj:] = eef_kp_his
    
    # get current state delta TODO: check
    eef_kp = np.stack(eef_kps[n_his-1:n_his+1], axis=0)  # (2, 1, 3)
    eef_kp_num = eef_kp.shape[1]
    states_delta = np.zeros((max_nobj + max_neef, obj_kp_start.shape[-1]), dtype=np.float32)
    states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]

    state_mask = np.zeros((max_nobj + max_neef), dtype=bool)
    state_mask[max_nobj : max_nobj + eef_kp_num] = True
    state_mask[:obj_kp_num] = True

    eef_mask = np.zeros((max_nobj + max_neef), dtype=bool)
    eef_mask[max_nobj : max_nobj + eef_kp_num] = True

    obj_mask = np.zeros((max_nobj,), dtype=bool)
    obj_mask[:obj_kp_num] = True
    
    ### visualize
    gt_kp_proj_last = []
    for cam in range(4):
        img_path = os.path.join(data_dir, f'episode_{episode_idx}', f'camera_{cam}', f'{start}_color.jpg')
        img = cv2.imread(img_path)
        intr = intr_list[cam]
        extr = extr_list[cam]
        save_dir_cam = os.path.join(save_dir, f'camera_{cam}')
        
        # transform keypoints
        kp_vis = state_history[-1, :obj_kp_num]
        obj_kp_homo = np.concatenate([kp_vis, np.ones((kp_vis.shape[0], 1))], axis=1) # (N, 3)
        obj_kp_homo = obj_kp_homo @ extr.T # (N, 3)
        
        obj_kp_homo[:, 1] *= -1
        obj_kp_homo[:, 2] *= -1
        
        # project keypoints
        fx, fy, cx, cy = intr
        obj_kp_proj = np.zeros((obj_kp_homo.shape[0], 2))
        obj_kp_proj[:, 0] = obj_kp_homo[:, 0] * fx / obj_kp_homo[:, 2] + cx
        obj_kp_proj[:, 1] = obj_kp_homo[:, 1] * fy / obj_kp_homo[:, 2] + cy
        
        # transform eef keypoints
        eef_kp_start = eef_kp[0]
        # print(f"eef_kp_start: {eef_kp_start}")
        eef_kp_homo = np.concatenate([eef_kp_start, np.ones((eef_kp_start.shape[0], 1))], axis=1) # (N, 3)
        eef_kp_homo = eef_kp_homo @ extr.T  # (N, 3)
        
        eef_kp_homo[:, 1] *= -1
        eef_kp_homo[:, 2] *= -1

        # also project eef keypoints
        fx, fy, cx, cy = intr
        eef_kp_proj = np.zeros((eef_kp_homo.shape[0], 2))
        eef_kp_proj[:, 0] = eef_kp_homo[:, 0] * fx / eef_kp_homo[:, 2] + cx
        eef_kp_proj[:, 1] = eef_kp_homo[:, 1] * fy / eef_kp_homo[:, 2] + cy
        
        # visualize points
        for k in range(obj_kp_proj.shape[0]):
            cv2.circle(img, (int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1])), point_size, 
                (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)
        
        # also visualize eef in red
        for k in range(eef_kp_proj.shape[0]):
            cv2.circle(img, (int(eef_kp_proj[k, 0]), int(eef_kp_proj[k, 1])), 3, 
                (0, 0, 255), -1)
        
        # starting pair visualization
        gt_kp_proj_last.append(obj_kp_proj)
        cv2.imwrite(os.path.join(save_dir_cam, f'{episode_idx:03}_{push_idx:02}_{start:03}_{end:03}_gt.jpg'), img)
        
    # iterative rollout
    current_start = start
    current_end = end
    idx_list = [[start, end]]
    for i in range(start_idx+1, start_idx+1+rollout_steps):
        # gt states
        gt_states = np.load(obj_kypts_paths).astype(np.float32)
        gt_state = gt_states[current_end]
        gt_state = [gt_state]
        gt_state = [gt_state[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
        gt_state = np.concatenate(gt_state, axis=0)
        # print(f"Frame {current_end} gt_state shape: {gt_state.shape}") # (max_obj, 3)
        gt_state = pad(gt_state, max_nobj)
        
        # next step input
        gt_kp = gt_state[obj_mask]
        # fps for viz
        gt_kp_vis = gt_kp[:obj_kp_num]
        
        # find next pair
        valid_pairs = pairs[pairs[:, n_his-1] == current_end]
        # avoid loop
        valid_pairs = valid_pairs[valid_pairs[:, n_his] > current_start]
        if len(valid_pairs) == 0:
            while current_end < n_frames:
                current_end += 1
                valid_pairs = pairs[pairs[:, n_his-1] == current_end]
                # avoid loop
                valid_pairs = valid_pairs[valid_pairs[:, n_his] > current_start]
                if len(valid_pairs) > 0:
                    break
            else:
                break
        next_pair = valid_pairs[int(len(valid_pairs) / 2)] # pick the middle one
        current_start = next_pair[n_his-1]
        current_end = next_pair[n_his]
        idx_list.append([current_start, current_end])
        # print(f"next pair: {current_start} {current_end}")
        
        # load eef keypoints
        eef_kps = np.load(eef_kypts_paths).astype(np.float32)
        eef_kp_start, eef_kp_end = eef_kps[current_start], eef_kps[current_end]
        eef_kp = np.stack([[eef_kp_start], [eef_kp_end]], axis=0)
        eef_kp_num = eef_kp.shape[1]
        # print(f"Frame {current_end} eef pos: {eef_kp_end}")
        eef_kp = pad(eef_kp, max_neef, dim=1)
        
        
                
        
    
if __name__ == '__main__':
    viz_gt()




