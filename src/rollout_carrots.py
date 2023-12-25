import glob
import numpy as np
import os
import sys
import torch
import json

from config.base_config import gen_args
from gnn.model import DynamicsPredictor
from gnn.utils import set_seed, umeyama_algorithm

import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import glob
from PIL import Image
import pickle as pkl
from dgl.geometry import farthest_point_sampler

from dataset.dataset_carrots import construct_edges_from_states, pad
from dataset.utils import extract_kp_single_frame, rgb_colormap, fps_rad_idx
from train_carrots import truncate_graph

def rollout_carrots(args, data_dir, prep_save_dir, save_dir, checkpoint, episode_idx, push_idx, start_idx, rollout_steps, colormap=None, vis=False, evaluate=False):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # data preparation hyperparameters
    n_future = 3
    n_his = 4

    # dist_thresh = 0.05  # only used in the dataset
    adj_thresh = 0.3
    fps_radius = 0.10
    max_n = 1
    max_nobj = 100
    max_neef = 1
    max_nR = 500

    # load model
    ## set args
    # particle encoder
    args.attr_dim = 2  # object and end effector
    args.n_his = n_his
    args.state_dim = 0  # x, y, z
    args.offset_dim = 0
    args.action_dim = 3
    args.pstep = 3
    args.time_step = 1
    args.dt = 1. / 60.
    args.sequence_length = 4
    args.phys_dim = 2  # TODO
    args.density_dim = 0  # particle density

    # relation encoder
    args.rel_particle_dim = 0
    args.rel_attr_dim = 2  # no attribute
    args.rel_group_dim = 1  # sum of difference of group one-hot vector
    args.rel_distance_dim = 3  # no distance
    args.rel_density_dim = 0  # no density

    # rel canonical (not used)
    args.rel_canonical_distance_dim = 0
    args.rel_canonical_attr_dim = 0
    args.rel_canonical_thresh = 3 * adj_thresh

    args.rel_can_attr_dim = 0

    args.use_vae = False
    args.phys_encode = False

    # model
    model_kwargs = {}
    model_kwargs.update({
        "predict_rigid": False,
        "predict_non_rigid": True,
        "rigid_out_dim": 0,
        "non_rigid_out_dim": 3,
    })
    # args.use_rigid_loss = False

    model = DynamicsPredictor(args, verbose=True, **model_kwargs)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(checkpoint))
    mse_loss = torch.nn.MSELoss()
    loss_funcs = [mse_loss]

    # vis
    n_fps_vis = 20
    point_size = 5
    line_size = 2
    line_alpha = 0.5

    # load pairs
    # pairs = np.loadtxt(os.path.join(prep_save_dir, 'frame_pairs', f'{episode_idx}.txt')).astype(int)
    pairs = np.loadtxt(os.path.join(prep_save_dir, 'frame_pairs', f'{episode_idx}_{push_idx}.txt')).astype(int)
    if len(pairs.shape) == 1: raise Exception
    print(f'Found {len(pairs)} frame pairs for episode {episode_idx}')

    # load can
    # can_path = os.path.join(prep_save_dir, 'canonical_pos', f'{episode_idx}.npy')

    # load kypts
    n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
    obj_kypts_paths = os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy")
    eef_kypts_paths = os.path.join(data_dir, f"episode_{episode_idx}/eef_pos.npy")

    # load physics
    physics_range = np.loadtxt(os.path.join(prep_save_dir, 'phys_range.txt')).astype(np.float32)
    physics_range = physics_range[:, 2:4]
    physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property.json")
    assert os.path.join(data_dir, f"episode_{episode_idx}/property.json") == physics_path
    with open(physics_path) as f:
        properties = json.load(f)
    physics_param = np.array([
        # properties['particle_radius'],
        # properties['num_particles'],
        properties['granular_scale'],
        properties['num_granular'],
        # properties['distribution_r'],
        # properties['dynamic_friction'],
        # properties['granular_mass']
    ]).astype(np.float32)
    physics_param = (physics_param - physics_range[0]) / (physics_range[1] - physics_range[0] + 1e-6)  # normalize

    # load camera (only for visualization)
    if vis:
        intr_list = [None] * 4
        extr_list = [None] * 4
        for cam in range(4):
            os.makedirs(os.path.join(save_dir, f'camera_{cam}'), exist_ok=True)
            intr_list[cam] = np.load(os.path.join(data_dir, f"camera_intrinsic_params.npy"))[cam]
            extr_list[cam] = np.load(os.path.join(data_dir, f"camera_extrinsic_matrix.npy"))[cam]

    # get starting pair
    start_pair_idx = start_idx
    pair = pairs[start_pair_idx]
    # print(pair[:n_his+1])
    start = pair[n_his-1]
    end = pair[n_his]

    # get history keypoints
    obj_kps = []
    eef_kps = []
    for i in range(n_his+1):
        frame_idx = pair[i]
        obj_kp, eef_kp = extract_kp_single_frame(data_dir, episode_idx, frame_idx)
        obj_kps.append(obj_kp)
        eef_kps.append(eef_kp)

    obj_kp_start = obj_kps[n_his-1]
    instance_num = len(obj_kp_start)
    assert instance_num == 1, 'only support single object'

    fps_idx_list = []
    # can_pos = np.load(canonical_pos[episode_idx])  # (N,)
    for j in range(len(obj_kp_start)):
        # farthest point sampling
        particle_tensor = torch.from_numpy(obj_kp_start[j]).float()[None, ...]
        fps_idx_tensor = farthest_point_sampler(particle_tensor, max_nobj, start_idx=np.random.randint(0, obj_kp_start[j].shape[0]))[0]
        fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)

        # downsample to uniform radius
        downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
        _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
        fps_idx_2 = fps_idx_2.astype(int)
        fps_idx = fps_idx_1[fps_idx_2]
        # print(fps_idx_1.shape, fps_idx_1.max(), fps_idx_1.dtype, fps_idx_2.shape, fps_idx_2.max(), fps_idx_2.dtype)
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

    # get current state delta
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

    # construct instance information
    p_rigid = np.zeros(max_n, dtype=np.float32)  # carrots are nonrigid
    p_instance = np.zeros((max_nobj, max_n), dtype=np.float32)
    j_perm = np.random.permutation(instance_num)
    ptcl_cnt = 0
    # sanity check
    assert sum([len(fps_idx_list[j]) for j in range(len(fps_idx_list))]) == obj_kp_num
    # fill in p_instance
    for j in range(instance_num):
        p_instance[ptcl_cnt:ptcl_cnt + len(fps_idx_list[j_perm[j]]), j_perm[j]] = 1
        ptcl_cnt += len(fps_idx_list[j_perm[j]])

    # construct physics information
    physics_param = np.tile(physics_param, (max_nobj, 1))  # (N, phys_dim)

    # construct attributes
    attr_dim = 2
    assert attr_dim == args.attr_dim
    attrs = np.zeros((max_nobj + max_neef, attr_dim), dtype=np.float32)
    attrs[:obj_kp_num, 0] = 1.
    attrs[max_nobj : max_nobj + eef_kp_num, 1] = 1.

    # construct relations (density as hyperparameter)
    Rr, Rs = construct_edges_from_states(torch.tensor(state_history[-1]).unsqueeze(0), adj_thresh * 1.5, 
                                        mask=torch.tensor(state_mask).unsqueeze(0), 
                                        eef_mask=torch.tensor(eef_mask).unsqueeze(0),
                                        no_self_edge=True)
    Rr, Rs = Rr.squeeze(0).numpy(), Rs.squeeze(0).numpy()

    # action encoded as state_delta (only stored in eef keypoints)
    states_delta = np.zeros((max_nobj + max_neef, obj_kp_start.shape[-1]), dtype=np.float32)
    states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]

    Rr = np.pad(Rr, ((0, max_nR - Rr.shape[0]), (0, 0)), mode='constant')
    Rs = np.pad(Rs, ((0, max_nR - Rs.shape[0]), (0, 0)), mode='constant')

    # save graph
    graph = {
        # input information
        "state": state_history,  # (n_his, N+M, state_dim)
        "action": states_delta,  # (N+M, state_dim)

        # relation information
        "Rr": Rr,  # (n_rel, N+M)
        "Rs": Rs,  # (n_rel, N+M)

        # attr information
        "attrs": attrs,  # (N+M, attr_dim)
        "p_rigid": p_rigid,  # (n_instance,)
        "p_instance": p_instance,  # (N, n_instance)
        "physics_param": physics_param,  # (N, phys_dim)
        "state_mask": state_mask,  # (N+M,)
        "eef_mask": eef_mask,  # (N+M,)
        "obj_mask": obj_mask,  # (N,)
    }

    pred_kp_proj_last = []
    gt_kp_proj_last = []
    if vis:
        for cam in range(4):
            img_path = os.path.join(data_dir, f'episode_{episode_idx}', f'camera_{cam}', f'{start}_color.jpg')
            img = cv2.imread(img_path)
            intr = intr_list[cam]
            extr = extr_list[cam]
            save_dir_cam = os.path.join(save_dir, f'camera_{cam}')

            # transform keypoints
            kp_vis = state_history[-1, :obj_kp_num]
            obj_kp_homo = np.concatenate([kp_vis, np.ones((kp_vis.shape[0], 1))], axis=1) # (N, 4)
            obj_kp_homo = obj_kp_homo @ extr.T  # (N, 4)

            obj_kp_homo[:, 1] *= -1
            obj_kp_homo[:, 2] *= -1

            # project keypoints
            fx, fy, cx, cy = intr
            obj_kp_proj = np.zeros((obj_kp_homo.shape[0], 2))
            obj_kp_proj[:, 0] = obj_kp_homo[:, 0] * fx / obj_kp_homo[:, 2] + cx
            obj_kp_proj[:, 1] = obj_kp_homo[:, 1] * fy / obj_kp_homo[:, 2] + cy

            # also transform eef keypoints
            eef_kp_start = eef_kp[0]
            eef_kp_homo = np.concatenate([eef_kp_start, np.ones((eef_kp_start.shape[0], 1))], axis=1) # (N, 4)
            eef_kp_homo = eef_kp_homo @ extr.T  # (N, 4)

            eef_kp_homo[:, 1] *= -1
            eef_kp_homo[:, 2] *= -1

            # also project eef keypoints
            fx, fy, cx, cy = intr
            eef_kp_proj = np.zeros((eef_kp_homo.shape[0], 2))
            eef_kp_proj[:, 0] = eef_kp_homo[:, 0] * fx / eef_kp_homo[:, 2] + cx
            eef_kp_proj[:, 1] = eef_kp_homo[:, 1] * fy / eef_kp_homo[:, 2] + cy

            # visualize
            for k in range(obj_kp_proj.shape[0]):
                cv2.circle(img, (int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1])), point_size, 
                    (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)

            # also visualize eef in red
            for k in range(eef_kp_proj.shape[0]):
                cv2.circle(img, (int(eef_kp_proj[k, 0]), int(eef_kp_proj[k, 1])), 3, 
                    (0, 0, 255), -1)

            # visualize edges
            for k in range(Rr.shape[0]):
                if Rr[k].sum() == 0: continue
                receiver = Rr[k].argmax()
                sender = Rs[k].argmax()
                if receiver >= max_nobj:  # eef
                    cv2.line(img, 
                        (int(eef_kp_proj[receiver - max_nobj, 0]), int(eef_kp_proj[receiver - max_nobj, 1])), 
                        (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                        (0, 0, 255), 2)
                elif sender >= max_nobj:  # eef
                    cv2.line(img, 
                        (int(eef_kp_proj[sender - max_nobj, 0]), int(eef_kp_proj[sender - max_nobj, 1])), 
                        (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                        (0, 0, 255), 2)
                else:
                    cv2.line(img, 
                        (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                        (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                        (0, 255, 0), 2)

            pred_kp_proj_last.append(obj_kp_proj)
            gt_kp_proj_last.append(obj_kp_proj)
            
            cv2.imwrite(os.path.join(save_dir_cam, f'{episode_idx:06}_{push_idx:06}_{start:06}_{end:06}_pred.jpg'), img)
            cv2.imwrite(os.path.join(save_dir_cam, f'{episode_idx:06}_{push_idx:06}_{start:06}_{end:06}_gt.jpg'), img)
            img = np.concatenate([img, img], axis=1)
            cv2.imwrite(os.path.join(save_dir_cam, f'{episode_idx:06}_{push_idx:06}_{start:06}_{end:06}_both.jpg'), img)

    graph = {key: torch.from_numpy(graph[key]).unsqueeze(0).to(device) for key in graph.keys()}

    # iterative rollout
    if vis:
        gt_lineset = [[], [], [], []]
        pred_lineset = [[], [], [], []]
    error_list = []
    current_start = start
    current_end = end
    idx_list = [[start, end]]
    with torch.no_grad():
        for i in range(start_idx + 1, start_idx + 1 + rollout_steps):
            # show t_line steps of lineset
            if vis:
                t_line = 5
                gt_lineset_new = [[], [], [], []]
                pred_lineset_new = [[], [], [], []]
                for lc in range(4):
                    for li in range(len(gt_lineset[lc])):
                        if gt_lineset[lc][li][-1] >= i - t_line:
                            gt_lineset_new[lc].append(gt_lineset[lc][li])
                            pred_lineset_new[lc].append(pred_lineset[lc][li])
                gt_lineset = gt_lineset_new
                pred_lineset = pred_lineset_new

            graph = truncate_graph(graph)
            pred_state, pred_motion = model(**graph)
            pred_state = pred_state.detach().cpu().numpy()

            # prepare gt
            gt_states = np.load(obj_kypts_paths).astype(np.float32)
            gt_state = gt_states[current_end]
            gt_state = [gt_state]
            gt_state = [gt_state[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            gt_state = np.concatenate(gt_state, axis=0)
            gt_state = pad(gt_state, max_nobj)

            # next step input
            obj_kp = pred_state[0][obj_mask]
            gt_kp = gt_state[obj_mask]

            # fps for visualization
            obj_kp_vis = obj_kp[:obj_kp_num]
            gt_kp_vis = gt_kp[:obj_kp_num]

            # calculate error
            error = np.linalg.norm(gt_kp - obj_kp, axis=1).mean()
            error_list.append(error)

            # find next pair
            valid_pairs = pairs[pairs[:, n_his-1] == current_end]
            # avoid loop
            valid_pairs = valid_pairs[valid_pairs[:, n_his] > current_end]
            if len(valid_pairs) == 0:
                while current_end < n_frames:
                    current_end += 1
                    valid_pairs = pairs[pairs[:, n_his-1] == current_end]
                    # avoid loop
                    valid_pairs = valid_pairs[valid_pairs[:, n_his] > current_end]
                    if len(valid_pairs) > 0:
                        break
                else:
                    break
            next_pair = valid_pairs[int(len(valid_pairs)/2)]  # pick the middle one
            current_start = next_pair[n_his-1]
            current_end = next_pair[n_his]
            idx_list.append([current_start, current_end])

            # generate next graph
            # load eef kypts
            eef_kps = np.load(eef_kypts_paths).astype(np.float32)
            eef_kp_start, eef_kp_end = eef_kps[current_start], eef_kps[current_end]
            x_start = eef_kp_start[0]
            z_start = eef_kp_start[1]
            x_end = eef_kp_end[0]
            z_end = eef_kp_end[1]
            y = np.mean(obj_kp[:, 1])
            eef_kp = np.array([[[x_start, y, -z_start]], [[x_end, y, -z_end]]], dtype=np.float32)  # (2, 1, 3)
            eef_kp_num = eef_kp.shape[1]
            eef_kp = pad(eef_kp, max_neef, dim=1)

            states = np.concatenate([pred_state[0], eef_kp[0]], axis=0)
            Rr, Rs = construct_edges_from_states(torch.tensor(states).unsqueeze(0), adj_thresh * 1.5, 
                                                mask=torch.tensor(state_mask).unsqueeze(0), 
                                                eef_mask=torch.tensor(eef_mask).unsqueeze(0),
                                                no_self_edge=True)
            Rr, Rs = Rr.squeeze(0).numpy(), Rs.squeeze(0).numpy()
            Rr = np.pad(Rr, ((0, max_nR - Rr.shape[0]), (0, 0)), mode='constant')
            Rs = np.pad(Rs, ((0, max_nR - Rs.shape[0]), (0, 0)), mode='constant')

            # action encoded as state_delta (only stored in eef keypoints)
            states_delta = np.zeros((max_nobj + max_neef, states.shape[-1]), dtype=np.float32)
            states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]

            state_history = np.concatenate([state_history[1:], states[None]], axis=0)

            graph = {
                "state": torch.from_numpy(state_history).unsqueeze(0).to(device),  # (n_his, N+M, state_dim)
                "action": torch.from_numpy(states_delta).unsqueeze(0).to(device),  # (N+M, state_dim)
                
                "Rr": torch.from_numpy(Rr).unsqueeze(0).to(device),  # (n_rel, N+M)
                "Rs": torch.from_numpy(Rs).unsqueeze(0).to(device),  # (n_rel, N+M)
                
                "attrs": graph["attrs"],  # (N+M, attr_dim)
                "p_rigid": graph["p_rigid"],  # (n_instance,)
                "p_instance": graph["p_instance"],  # (N, n_instance)
                "physics_param": graph["physics_param"],  # (N,)
                "obj_mask": graph["obj_mask"],  # (N,)
            }

            # visualize
            if vis:
                pred_kp_proj_list = []
                gt_kp_proj_list = []
                for cam in range(4):
                    img_path = os.path.join(data_dir, f'episode_{episode_idx}', f'camera_{cam}', f'{current_start}_color.jpg')
                    img_orig = cv2.imread(img_path)
                    img = img_orig.copy()
                    intr = intr_list[cam]
                    extr = extr_list[cam]
                    save_dir_cam = os.path.join(save_dir, f'camera_{cam}')

                    # transform keypoints
                    obj_kp_homo = np.concatenate([obj_kp_vis, np.ones((obj_kp_vis.shape[0], 1))], axis=1) # (N, 4)
                    obj_kp_homo = obj_kp_homo @ extr.T  # (N, 4)

                    obj_kp_homo[:, 1] *= -1
                    obj_kp_homo[:, 2] *= -1

                    # project keypoints
                    fx, fy, cx, cy = intr
                    obj_kp_proj = np.zeros((obj_kp_homo.shape[0], 2))
                    obj_kp_proj[:, 0] = obj_kp_homo[:, 0] * fx / obj_kp_homo[:, 2] + cx
                    obj_kp_proj[:, 1] = obj_kp_homo[:, 1] * fy / obj_kp_homo[:, 2] + cy

                    pred_kp_proj_list.append(obj_kp_proj)

                    # also transform eef keypoints
                    eef_kp_vis = eef_kp[0, :eef_kp_num]
                    eef_kp_homo = np.concatenate([eef_kp_vis, np.ones((eef_kp_vis.shape[0], 1))], axis=1) # (N, 4)
                    eef_kp_homo = eef_kp_homo @ extr.T  # (N, 4)

                    eef_kp_homo[:, 1] *= -1
                    eef_kp_homo[:, 2] *= -1

                    # also project eef keypoints
                    fx, fy, cx, cy = intr
                    eef_kp_proj = np.zeros((eef_kp_homo.shape[0], 2))
                    eef_kp_proj[:, 0] = eef_kp_homo[:, 0] * fx / eef_kp_homo[:, 2] + cx
                    eef_kp_proj[:, 1] = eef_kp_homo[:, 1] * fy / eef_kp_homo[:, 2] + cy

                    # visualize
                    for k in range(obj_kp_proj.shape[0]):
                        cv2.circle(img, (int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1])), point_size, 
                            (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)

                    # also visualize eef in red
                    for k in range(eef_kp_proj.shape[0]):
                        cv2.circle(img, (int(eef_kp_proj[k, 0]), int(eef_kp_proj[k, 1])), point_size, 
                            (0, 0, 255), -1)

                    pred_kp_last = pred_kp_proj_last[cam]
                    for k in range(obj_kp_proj.shape[0]):
                        pred_lineset[cam].append([int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1]), int(pred_kp_last[k, 0]), int(pred_kp_last[k, 1]), 
                                            int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0]), i])
                    
                    # visualize edges
                    for k in range(Rr.shape[0]):
                        if Rr[k].sum() == 0: continue
                        receiver = Rr[k].argmax()
                        sender = Rs[k].argmax()
                        if receiver >= max_nobj:  # eef
                            cv2.line(img, 
                                (int(eef_kp_proj[receiver - max_nobj, 0]), int(eef_kp_proj[receiver - max_nobj, 1])), 
                                (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                                (0, 0, 255), 2)
                        elif sender >= max_nobj:  # eef
                            cv2.line(img, 
                                (int(eef_kp_proj[sender - max_nobj, 0]), int(eef_kp_proj[sender - max_nobj, 1])), 
                                (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                                (0, 0, 255), 2)
                        else:
                            cv2.line(img, 
                                (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                                (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                                (0, 255, 0), 2)

                    img_overlay = img.copy()
                    for k in range(len(pred_lineset[cam])):
                        ln = pred_lineset[cam][k]
                        cv2.line(img_overlay, (ln[0], ln[1]), (ln[2], ln[3]), (ln[4], ln[5], ln[6]), line_size)

                    cv2.addWeighted(img_overlay, line_alpha, img, 1 - line_alpha, 0, img)
                    cv2.imwrite(os.path.join(save_dir_cam, f'{episode_idx:06}_{push_idx:06}_{current_start:06}_{current_end:06}_pred.jpg'), img)
                    img_pred = img.copy()

                    # visualize gt similarly
                    img = img_orig.copy()
                    gt_kp_homo = np.concatenate([gt_kp_vis, np.ones((gt_kp_vis.shape[0], 1))], axis=1) # (N, 4)
                    gt_kp_homo = gt_kp_homo @ extr.T  # (N, 4)
                    gt_kp_homo[:, 1] *= -1
                    gt_kp_homo[:, 2] *= -1        
                    gt_kp_proj = np.zeros((gt_kp_homo.shape[0], 2))
                    gt_kp_proj[:, 0] = gt_kp_homo[:, 0] * fx / gt_kp_homo[:, 2] + cx
                    gt_kp_proj[:, 1] = gt_kp_homo[:, 1] * fy / gt_kp_homo[:, 2] + cy

                    gt_kp_proj_list.append(gt_kp_proj)
                    
                    for k in range(gt_kp_proj.shape[0]):
                        cv2.circle(img, (int(gt_kp_proj[k, 0]), int(gt_kp_proj[k, 1])), point_size, 
                            (int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0])), -1)

                    gt_kp_last = gt_kp_proj_last[cam]
                    for k in range(gt_kp_proj.shape[0]):
                        gt_lineset[cam].append([int(gt_kp_proj[k, 0]), int(gt_kp_proj[k, 1]), int(gt_kp_last[k, 0]), int(gt_kp_last[k, 1]), 
                                        int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0]), i])

                    # also visualize eef in red
                    for k in range(eef_kp_proj.shape[0]):
                        cv2.circle(img, (int(eef_kp_proj[k, 0]), int(eef_kp_proj[k, 1])), point_size, 
                            (0, 0, 255), -1)

                    # visualize edges
                    # for k in range(Rr.shape[0]):
                    #     if Rr[k].sum() == 0: continue
                    #     receiver = Rr[k].argmax()
                    #     sender = Rs[k].argmax()
                    #     if receiver >= max_nobj:  # eef
                    #         cv2.line(img, 
                    #             (int(eef_kp_proj[receiver - max_nobj, 0]), int(eef_kp_proj[receiver - max_nobj, 1])), 
                    #             (int(gt_kp_proj[sender, 0]), int(gt_kp_proj[sender, 1])), 
                    #             (0, 0, 255), 2)
                    #     elif sender >= max_nobj:  # eef
                    #         cv2.line(img, 
                    #             (int(eef_kp_proj[sender - max_nobj, 0]), int(eef_kp_proj[sender - max_nobj, 1])), 
                    #             (int(gt_kp_proj[receiver, 0]), int(gt_kp_proj[receiver, 1])), 
                    #             (0, 0, 255), 2)
                    #     else:
                    #         cv2.line(img, 
                    #             (int(gt_kp_proj[receiver, 0]), int(gt_kp_proj[receiver, 1])), 
                    #             (int(gt_kp_proj[sender, 0]), int(gt_kp_proj[sender, 1])), 
                    #             (0, 255, 0), 2)

                    img_overlay = img.copy()
                    for k in range(len(gt_lineset[cam])):
                        ln = gt_lineset[cam][k]
                        cv2.line(img_overlay, (ln[0], ln[1]), (ln[2], ln[3]), (ln[4], ln[5], ln[6]), line_size)

                    cv2.imwrite(os.path.join(save_dir_cam, f'{episode_idx:06}_{push_idx:06}_{current_start:06}_{current_end:06}_gt.jpg'), img)
                    img_gt = img.copy()

                    img = np.concatenate([img_pred, img_gt], axis=1)
                    cv2.imwrite(os.path.join(save_dir_cam, f'{episode_idx:06}_{push_idx:06}_{current_start:06}_{current_end:06}_both.jpg'), img)

                pred_kp_proj_last = pred_kp_proj_list
                gt_kp_proj_last = gt_kp_proj_list
    
    # idx_list = np.array(idx_list)
    # print(idx_list.shape)

    if vis:
        # plot error
        plt.figure(figsize=(10, 5))
        plt.plot(error_list)
        plt.xlabel("time step")
        plt.ylabel("error")
        plt.grid()
        plt.savefig(os.path.join(save_dir, 'error.png'), dpi=300)
        plt.close()

    # save error
    if evaluate:
        error_list = np.array(error_list)
        np.savetxt(os.path.join(save_dir, f'error_{episode_idx}-{push_idx}-{start_idx}-{rollout_steps}.txt'), error_list)
        return error_list

def rollout_vis():
    args = gen_args()

    data_name = "carrots_1"

    data_dir = f"/mnt/sda/data/{data_name}"

    episode_idx = 24
    push_idx = 0
    start_idx = 0
    rollout_steps = 100

    checkpoint_dir_name = "carrots_1_2"

    checkpoint_epoch = 100
    checkpoint = f"/mnt/sda/relation_logs/{checkpoint_dir_name}/checkpoints/model_{checkpoint_epoch}.pth"

    prep_save_dir = f"/mnt/sda/preprocess/{data_name}"

    colormap = rgb_colormap(repeat=100)  # only red
    # colormap = label_colormap()

    save_dir = f"/mnt/sda/vis/rollout-vis-{checkpoint_dir_name}-model_{checkpoint_epoch}-{data_dir.split('/')[-1]}-{episode_idx}-{start_idx}-{rollout_steps}"# -{dense_str}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"rollout {episode_idx} from {start_idx} to {start_idx + rollout_steps} with {checkpoint}")
    print(f"saving to {save_dir}")

    rollout_carrots(args, data_dir, prep_save_dir, save_dir, checkpoint, episode_idx, push_idx, start_idx, rollout_steps, colormap, vis=True, evaluate=False)

    for cam in range(4):
        img_path = os.path.join(save_dir, f"camera_{cam}")
        frame_rate = 4
        height = 360
        width = 640
        pred_out_path = os.path.join(img_path, "pred.mp4")
        os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/{episode_idx:06}_*_pred.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {pred_out_path} -y")
        gt_out_path = os.path.join(img_path, "gt.mp4")
        os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/{episode_idx:06}_*_gt.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {gt_out_path} -y")
        both_out_path = os.path.join(img_path, "both.mp4")
        os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/{episode_idx:06}_*_both.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {both_out_path} -y")

def rollout_eval():
    args = gen_args()

    data_name = "carrots_1"

    data_dir = f"/mnt/sda/data/{data_name}"

    checkpoint_dir_name = "carrots_1_5"

    checkpoint_epoch = 100
    checkpoint = f"/mnt/sda/relation_logs/{checkpoint_dir_name}/checkpoints/model_{checkpoint_epoch}.pth"

    prep_save_dir = f"/mnt/sda/preprocess/{data_name}"

    save_dir = f"/mnt/sda/eval/rollout-eval-{checkpoint_dir_name}-model_{checkpoint_epoch}-{data_dir.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)
    
    episode_range = range(22, 25)
    push_range = range(0, 5)
    start_idx = 0
    rollout_steps = 100

    total_error = []
    for episode_idx in episode_range:
        for push_idx in push_range:

            print(f"rollout {episode_idx} from {start_idx} to {start_idx + rollout_steps} with {checkpoint}")
            print(f"saving to {save_dir}")

            error = rollout_carrots(args, data_dir, prep_save_dir, save_dir, checkpoint, episode_idx, push_idx, start_idx, rollout_steps, vis=False, evaluate=True)
            total_error.append(error)
    
    max_step = max([len(total_error[i]) for i in range(len(total_error))])
    min_step = min([len(total_error[i]) for i in range(len(total_error))])
    step_error = np.zeros((min_step, len(total_error)))
    for i in range(min_step):
        for j in range(len(total_error)):
            # step_error[i] = np.mean([total_error[j][i] for j in range(len(total_error)) if i < len(total_error[j])])
            step_error[i, j] = total_error[j][i]
    
    # step_mean_error = step_error.mean(1)
    np.savetxt(os.path.join(save_dir, "error.txt"), step_error)

    # Calculate the median, 75th percentile, and 25th percentile
    median_error = np.median(step_error, axis=1)
    step_75_error = np.percentile(step_error, 75, axis=1)
    step_25_error = np.percentile(step_error, 25, axis=1)

    # plot error
    plt.figure(figsize=(10, 5))
    plt.plot(median_error)
    plt.xlabel("time step")
    plt.ylabel("error")
    plt.grid()

    ax = plt.gca()
    x = np.arange(median_error.shape[0])
    ax.fill_between(x, step_25_error, step_75_error, alpha=0.2)

    plt.savefig(os.path.join(save_dir, 'error.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    rollout_vis()
    rollout_eval()