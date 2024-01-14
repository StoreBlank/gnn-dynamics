import glob
import numpy as np
import os
import sys
import torch
import json
import argparse
import yaml

from gnn.model import DynamicsPredictor
from gnn.utils import set_seed, umeyama_algorithm

import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import glob
from dgl.geometry import farthest_point_sampler

from dataset import construct_edges_from_states, load_dataset
from utils import rgb_colormap, fps_rad_idx, pad, vis_points, moviepy_merge_video
from train import truncate_graph


# component functions for rollout
def visualize_graph(data_dir, episode_idx, start, end, vis_t, save_dir,
        kp_vis, gt_kp_vis, tool_kp, Rr, Rs,
        max_nobj,
        colormap=None, point_size=4, edge_size=1, line_size=2, line_alpha=0.5, t_line=5,
        gt_lineset=None, pred_lineset=None, 
        pred_kp_proj_last=None, gt_kp_proj_last=None):
    
    if colormap is None:
        colormap = rgb_colormap(repeat=100)
    
    if pred_kp_proj_last is None:
        assert gt_kp_proj_last is None
        pred_kp_proj_last = [None, None, None, None]
        gt_kp_proj_last = [None, None, None, None]
    else:
        assert gt_kp_proj_last is not None
    
    # temporary
    pred_kp_proj_list = []
    gt_kp_proj_list = []

    if gt_lineset is None:
        assert pred_lineset is None
        gt_lineset = [[], [], [], []]
        pred_lineset = [[], [], [], []]
    else:
        assert pred_lineset is not None
        gt_lineset_new = [[], [], [], []]
        pred_lineset_new = [[], [], [], []]
        for lc in range(4):
            for li in range(len(gt_lineset[lc])):
                if gt_lineset[lc][li][-1] >= vis_t - t_line:
                    gt_lineset_new[lc].append(gt_lineset[lc][li])
                    pred_lineset_new[lc].append(pred_lineset[lc][li])
        gt_lineset = gt_lineset_new
        pred_lineset = pred_lineset_new

    for cam in range(4):
        intr = np.load(os.path.join(data_dir, f"camera_intrinsic_params.npy"))[cam]
        extr = np.load(os.path.join(data_dir, f"camera_extrinsic_matrix.npy"))[cam]

        img_path = os.path.join(data_dir, f'episode_{episode_idx}', f'camera_{cam}', f'{start}_color.jpg')
        img_orig = cv2.imread(img_path)
        img = img_orig.copy()
        # intr = intr_list[cam]
        # extr = extr_list[cam]
        save_dir_cam = os.path.join(save_dir, f'camera_{cam}')
        os.makedirs(save_dir_cam, exist_ok=True)

        # visualize keypoints
        obj_kp_proj, img = vis_points(kp_vis, intr, extr, img, point_size=point_size, point_color=(255, 0, 0)) # blue
        pred_kp_proj_list.append(obj_kp_proj)
        
        tool_kp_vis = tool_kp[0]
        tool_kp_proj, img = vis_points(tool_kp_vis, intr, extr, img, point_size=point_size, point_color=(0, 0, 255)) # red

        # visualize lineset
        pred_kp_last = pred_kp_proj_last[cam]
        if not (pred_kp_last is None):
            for k in range(obj_kp_proj.shape[0]):
                pred_lineset[cam].append([int(obj_kp_proj[k, 0]), int(obj_kp_proj[k, 1]), int(pred_kp_last[k, 0]), int(pred_kp_last[k, 1]), 
                                    int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0]), vis_t])

        # visualize edges
        for k in range(Rr.shape[0]):
            if Rr[k].sum() == 0: continue
            receiver = Rr[k].argmax()
            sender = Rs[k].argmax()
            if receiver >= max_nobj:  # tool
                cv2.line(img, 
                    (int(tool_kp_proj[receiver - max_nobj, 0]), int(tool_kp_proj[receiver - max_nobj, 1])), 
                    (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                    (0, 0, 255), edge_size)
            elif sender >= max_nobj:  # tool
                cv2.line(img, 
                    (int(tool_kp_proj[sender - max_nobj, 0]), int(tool_kp_proj[sender - max_nobj, 1])), 
                    (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                    (0, 0, 255), edge_size)
            else:
                cv2.line(img, 
                    (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                    (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                    (0, 255, 0), edge_size)

        # overlay lineset
        img_overlay = img.copy()
        for k in range(len(pred_lineset[cam])):
            ln = pred_lineset[cam][k]
            cv2.line(img_overlay, (ln[0], ln[1]), (ln[2], ln[3]), (ln[4], ln[5], ln[6]), line_size)

        cv2.addWeighted(img_overlay, line_alpha, img, 1 - line_alpha, 0, img)
        cv2.imwrite(os.path.join(save_dir_cam, f'{start:06}_{end:06}_pred.jpg'), img)
        img_pred = img.copy()

        # visualize gt similarly
        img = img_orig.copy()

        # visualize keypoints
        gt_kp_proj, img = vis_points(gt_kp_vis, intr, extr, img, point_size=point_size, point_color=(255, 0, 0)) # blue
        gt_kp_proj_list.append(gt_kp_proj)
    
        tool_kp_vis = tool_kp[0]
        tool_kp_proj, img = vis_points(tool_kp_vis, intr, extr, img, point_size=point_size, point_color=(0, 0, 255)) # red

        # visualize lineset
        gt_kp_last = gt_kp_proj_last[cam]
        if not (gt_kp_last is None):
            for k in range(gt_kp_proj.shape[0]):
                gt_lineset[cam].append([int(gt_kp_proj[k, 0]), int(gt_kp_proj[k, 1]), int(gt_kp_last[k, 0]), int(gt_kp_last[k, 1]), 
                                int(colormap[k, 2]), int(colormap[k, 1]), int(colormap[k, 0]), vis_t])

        # visualize edges (for gt, edges will not reflect adjacency)
        for k in range(Rr.shape[0]):
            if Rr[k].sum() == 0: continue
            receiver = Rr[k].argmax()
            sender = Rs[k].argmax()
            if receiver >= max_nobj:  # tool
                cv2.line(img, 
                    (int(tool_kp_proj[receiver - max_nobj, 0]), int(tool_kp_proj[receiver - max_nobj, 1])), 
                    (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                    (0, 0, 255), edge_size)
            elif sender >= max_nobj:  # tool
                cv2.line(img, 
                    (int(tool_kp_proj[sender - max_nobj, 0]), int(tool_kp_proj[sender - max_nobj, 1])), 
                    (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                    (0, 0, 255), edge_size)
            else:
                cv2.line(img, 
                    (int(obj_kp_proj[receiver, 0]), int(obj_kp_proj[receiver, 1])), 
                    (int(obj_kp_proj[sender, 0]), int(obj_kp_proj[sender, 1])), 
                    (0, 255, 0), edge_size)

        img_overlay = img.copy()
        for k in range(len(gt_lineset[cam])):
            ln = gt_lineset[cam][k]
            cv2.line(img_overlay, (ln[0], ln[1]), (ln[2], ln[3]), (ln[4], ln[5], ln[6]), line_size)

        cv2.imwrite(os.path.join(save_dir_cam, f'{start:06}_{end:06}_gt.jpg'), img)
        img_gt = img.copy()

        img = np.concatenate([img_pred, img_gt], axis=1)
        cv2.imwrite(os.path.join(save_dir_cam, f'{start:06}_{end:06}_both.jpg'), img)
    
    pred_kp_proj_last = pred_kp_proj_list
    gt_kp_proj_last = gt_kp_proj_list
    
    return pred_kp_proj_last, gt_kp_proj_last, gt_lineset, pred_lineset


# component functions for rollout
def construct_graph(dataset, n_his, pair, episode_idx, physics_param, material_config, all_particles_pos, all_tool_states):
    max_n = dataset['max_n']
    max_tool = dataset['max_tool']
    max_nobj = dataset['max_nobj']
    max_ntool = dataset['max_ntool']
    max_nR = dataset['max_nR']
    fps_radius = (dataset['fps_radius_range'][0] + dataset['fps_radius_range'][1]) / 2
    adj_thresh = (dataset['adj_radius_range'][0] + dataset['adj_radius_range'][1]) / 2
    
    ### construct graph ###

    # get history keypoints
    obj_kps, tool_kps = [], []
    for i in range(len(pair)):
        frame_idx = pair[i]
        # obj_kp, tool_kp = extract_kp_single_frame(dataset["data_dir"], episode_idx, frame_idx)
        obj_kp = all_particles_pos[episode_idx][frame_idx][None] # (1, num_obj_points, 3)
        tool_kp = all_tool_states[episode_idx][frame_idx] # (num_tool_points, 3)

        obj_kps.append(obj_kp)
        tool_kps.append(tool_kp)

    obj_kp_start = obj_kps[n_his-1]
    instance_num = len(obj_kp_start)
    assert instance_num == 1, 'only support single object'

    fps_idx_list = []
    # can_pos = np.load(canonical_pos[episode_idx])  # (N,)
    # print(f"obj_kp_start: {obj_kp_start.shape}")
    print(f"obj_kp_start: min {obj_kp_start[0].min(0)}, max {obj_kp_start[0].max(0)}")
    for j in range(len(obj_kp_start)):
        # farthest point sampling
        particle_tensor = torch.from_numpy(obj_kp_start[j]).float()[None, ...]
        fps_idx_tensor = farthest_point_sampler(particle_tensor, max_nobj, start_idx=np.random.randint(0, obj_kp_start[j].shape[0]))[0]
        fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)

        # downsample to uniform radius
        downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
        _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
        fps_idx_2 = fps_idx_2.astype(int)
        print(f"fps_idx_2: {fps_idx_2.shape}")
        fps_idx = fps_idx_1[fps_idx_2]
        fps_idx_list.append(fps_idx)

    # downsample to get current obj kp
    obj_kp_start = [obj_kp_start[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
    obj_kp_start = np.concatenate(obj_kp_start, axis=0) # (N, 3)
    obj_kp_num = obj_kp_start.shape[0]
    # note: obj_kp_start is not used after this point

    # load history states
    state_history = np.zeros((n_his, max_nobj + max_ntool * max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
    for fi in range(n_his):
        obj_kp_his = obj_kps[fi]
        obj_kp_his = [obj_kp_his[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
        obj_kp_his = np.concatenate(obj_kp_his, axis=0)
        obj_kp_his = pad(obj_kp_his, max_nobj)
        state_history[fi, :max_nobj] = obj_kp_his

        tool_kp_his = tool_kps[fi]
        tool_kp_his = pad(tool_kp_his, max_ntool * max_tool)
        state_history[fi, max_nobj : max_nobj + max_ntool * max_tool] = tool_kp_his

    # get current state delta
    tool_kp = np.stack(tool_kps[n_his-1:n_his+1], axis=0)  # (2, N, 3)
    tool_kp_num = tool_kp.shape[1]

    states_delta = np.zeros((max_nobj + max_ntool * max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
    states_delta[max_nobj : max_nobj + tool_kp_num] = tool_kp[1] - tool_kp[0]

    # get masks
    state_mask = np.zeros((max_nobj + max_ntool * max_tool), dtype=bool)
    state_mask[:obj_kp_num] = True # obj
    state_mask[max_nobj : max_nobj + tool_kp_num] = True
    
    obj_mask = np.zeros((max_nobj,), dtype=bool)
    obj_mask[:obj_kp_num] = True
    
    tool_mask = np.zeros((max_nobj + max_ntool * max_tool,), dtype=bool)
    tool_mask[max_nobj : max_nobj + tool_kp_num] = True # dynamic tool

    # construct instance information
    p_rigid = np.zeros(max_n, dtype=np.float32)  # clothes are nonrigid
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
    for material_name in dataset['materials']:
        if material_name not in physics_param.keys():
            raise ValueError(f'Physics parameter {material_name} not found in {dataset["data_dir"]}')
        # physics_param[material_name] += np.random.uniform(-phys_noise, phys_noise, 
        #         size=physics_param[material_name].shape)

    # new: construct physics information for each particle
    material_idx = np.zeros((max_nobj, len(material_config['material_index'])), dtype=np.int32)
    assert len(dataset['materials']) == 1, 'only support single material'
    material_idx[:obj_kp_num, material_config['material_index'][dataset['materials'][0]]] = 1

    # construct attributes
    attr_dim = 2
    attrs = np.zeros((max_nobj + max_ntool * max_tool, attr_dim), dtype=np.float32)
    attrs[:obj_kp_num, 0] = 1.
    attrs[max_nobj : max_nobj + tool_kp_num, 1] = 1

    # numpy to torch
    state_history = torch.from_numpy(state_history).float()
    states_delta = torch.from_numpy(states_delta).float()
    attrs = torch.from_numpy(attrs).float()
    p_rigid = torch.from_numpy(p_rigid).float()
    p_instance = torch.from_numpy(p_instance).float()
    physics_param = {k: torch.from_numpy(v).float() for k, v in physics_param.items()}
    material_idx = torch.from_numpy(material_idx).long()
    state_mask = torch.from_numpy(state_mask)
    tool_mask = torch.from_numpy(tool_mask)
    obj_mask = torch.from_numpy(obj_mask)
    tool_kp = torch.from_numpy(tool_kp).float()

    # construct relations (density as hyperparameter)
    Rr, Rs = construct_edges_from_states(state_history[-1][None], adj_thresh, 
                mask=state_mask[None], tool_mask=tool_mask[None], no_self_edge=True)
    # print(f"Rr: {Rr.shape}, Rs: {Rs.shape}")
    Rr = Rr[0].numpy()
    Rs = Rs[0].numpy()
    Rr = pad(Rr, max_nR)
    Rs = pad(Rs, max_nR)
    Rr = torch.from_numpy(Rr).float()
    Rs = torch.from_numpy(Rs).float()

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
        # "physics_param": physics_param,  # (N, phys_dim)
        "state_mask": state_mask,  # (N+M,)
        "tool_mask": tool_mask,  # (N+M,)
        "obj_mask": obj_mask,  # (N,)

        "material_index": material_idx,  # (N, num_materials)

        # for non-model use
        "tool_kp": tool_kp,  # (2, max_ntool * max_tool, 3)
    }

    for material_name in physics_param.keys():
        graph[material_name + '_physics_param'] = physics_param[material_name]

    ### finish constructing graph ###
    return graph, fps_idx_list


# component functions for rollout
def get_next_pair_or_break_episode(pairs, n_his, n_frames, current_end):
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
            return None
    next_pair = valid_pairs[int(len(valid_pairs)/2)]  # pick the middle one
    return next_pair


# component functions for rollout
def get_next_pair_or_break_episode_pushes(pairs, n_his, n_frames, current_end):
    # find next pair
    valid_pairs = pairs[pairs[:, n_his-1] == current_end]
    # avoid loop
    valid_pairs = valid_pairs[valid_pairs[:, n_his] > current_end]
    if len(valid_pairs) == 0:
        return None
    next_pair = valid_pairs[int(len(valid_pairs)/2)]  # pick the middle one
    return next_pair


def rollout_from_start_graph(graph, model, material_config, device, dataset, episode_idx, current_start, current_end, 
        get_next_pair_or_break_func, fps_idx_list, pairs, save_dir, all_particles_pos, all_tool_states):

    obj_mask = graph['obj_mask'].numpy()
    obj_kp_num = obj_mask.sum()
    print(f"obj_kp_num: {obj_kp_num}")
    max_nobj = dataset['max_nobj']
    max_ntool = dataset['max_ntool']

    vis = True
    if vis:
        Rr = graph['Rr'].numpy()
        Rs = graph['Rs'].numpy()
        tool_kp = graph['tool_kp'].numpy()
        kp_vis = graph['state'][-1, :obj_kp_num].numpy()
        pred_kp_proj_last, gt_kp_proj_last, gt_lineset, pred_lineset = \
            visualize_graph(dataset['data_dir'], episode_idx, current_start, current_end, 0, save_dir,
            kp_vis, kp_vis, tool_kp, Rr, Rs, max_nobj)

    graph = {key: graph[key].unsqueeze(0).to(device) for key in graph.keys()}

    # iterative rollout
    rollout_steps = 100
    error_list = []
    idx_list = [[current_start, current_end]]
    with torch.no_grad():
        for i in range(1, 1 + rollout_steps):
            # n_frames = np.load(os.path.join(dataset["data_dir"], f"episode_{episode_idx}/particles_pos.npy")).shape[1]
            n_frames = len(list(glob.glob(os.path.join(dataset['data_dir'], f"episode_{episode_idx}/camera_0/*_color.jpg"))))

            n_his = model.model_config['n_his']
            max_nobj = dataset['max_nobj']
            max_tool = dataset['max_tool']
            max_ntool = dataset['max_ntool']
            max_nR = dataset['max_nR']
            adj_thresh = (dataset['adj_radius_range'][0] + dataset['adj_radius_range'][1]) / 2

            graph = truncate_graph(graph)
            pred_state, pred_motion = model(**graph)
            pred_state = pred_state.detach().cpu().numpy()

            # prepare gt
            # gt_state, _ = extract_kp_single_frame(dataset["data_dir"], episode_idx, current_end)
            # gt_state = [gt_state]
            gt_state = all_particles_pos[episode_idx][current_end][None] # (1, num_obj_points, 3)
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

            next_pair = get_next_pair_or_break_func(pairs, n_his, n_frames, current_end)
            if next_pair is None:
                break
            current_start = next_pair[n_his-1]
            current_end = next_pair[n_his]
            idx_list.append([current_start, current_end])

            # generate next graph
            # load tool kypts
            tool_kp_start = all_tool_states[episode_idx][current_start] # (num_tool_points, 3)
            tool_kp_end = all_tool_states[episode_idx][current_end] # (num_tool_points, 3)

            tool_kp_num = tool_kp_start.shape[0]

            tool_kp = np.stack([tool_kp_start, tool_kp_end], axis=0)  # (2, N, 3)
            tool_kp = pad(tool_kp, max_ntool * max_tool, dim=1)

            states = np.concatenate([pred_state, tool_kp[0:1]], axis=1)
            assert states.shape[1] == max_nobj + max_ntool * max_tool
            assert states.shape[0] == 1
            Rr, Rs = construct_edges_from_states(torch.tensor(states), adj_thresh, 
                                                mask=graph['state_mask'], 
                                                tool_mask=graph['tool_mask'],
                                                no_self_edge=True)
            Rr = Rr[0].numpy()
            Rs = Rs[0].numpy()
            Rr = pad(Rr, max_nR)
            Rs = pad(Rs, max_nR)
            Rr = torch.from_numpy(Rr).float()
            Rs = torch.from_numpy(Rs).float()

            # action encoded as state_delta (only stored in tool keypoints)
            states_delta = np.zeros((max_nobj + max_ntool * max_tool, states.shape[-1]), dtype=np.float32)
            states_delta[max_nobj : max_nobj + tool_kp_num] = tool_kp[1] - tool_kp[0]
 
            state_history = graph['state'][0].detach().cpu().numpy()
            state_history = np.concatenate([state_history[1:], states], axis=0)

            new_graph = {
                "state": torch.from_numpy(state_history).unsqueeze(0).to(device),  # (n_his, N+M, state_dim)
                "action": torch.from_numpy(states_delta).unsqueeze(0).to(device),  # (N+M, state_dim)
                
                "Rr": Rr.unsqueeze(0).to(device),  # (n_rel, N+M)
                "Rs": Rs.unsqueeze(0).to(device),  # (n_rel, N+M)
                
                "attrs": graph["attrs"],  # (N+M, attr_dim)
                "p_rigid": graph["p_rigid"],  # (n_instance,)
                "p_instance": graph["p_instance"],  # (N, n_instance)
                # "physics_param": graph["physics_param"],
                "obj_mask": graph["obj_mask"],  # (N,)
                "tool_mask": graph["tool_mask"],  # (N+M,)
                "state_mask": graph["state_mask"],  # (N+M,)
                "material_index": graph["material_index"],  # (N, num_materials)
            }
            for name in graph.keys():
                if name.endswith('_physics_param'):
                    new_graph[name] = graph[name]
            
            graph = new_graph

            # visualize
            if vis:
                pred_kp_proj_last, gt_kp_proj_last, gt_lineset, pred_lineset = \
                    visualize_graph(dataset['data_dir'], episode_idx, current_start, current_end, i, save_dir,
                    obj_kp_vis, gt_kp_vis, tool_kp, Rr, Rs, max_nobj,
                    gt_lineset=gt_lineset, pred_lineset=pred_lineset,
                    pred_kp_proj_last=pred_kp_proj_last, gt_kp_proj_last=gt_kp_proj_last)

    return error_list


def rollout_episode(model, device, dataset, material_config, pairs, episode_idx, physics_param, save_dir, all_particles_pos, all_tool_states):
    n_his = model.model_config['n_his']

    # state_noise = 0.0
    # phys_noise = 0.0
    
    # get starting pair
    start_pair_idx = 0
    pair = pairs[start_pair_idx]
    start = pair[n_his-1]
    end = pair[n_his]

    graph, fps_idx_list = construct_graph(dataset, n_his, pair, episode_idx, physics_param, material_config, all_particles_pos, all_tool_states)
    
    error_list = rollout_from_start_graph(graph, model, material_config, device, dataset, episode_idx, start, end, 
            get_next_pair_or_break_episode, fps_idx_list, pairs, save_dir, all_particles_pos, all_tool_states)

    # plot error
    plt.figure(figsize=(10, 5))
    plt.plot(error_list)
    plt.xlabel("time step")
    plt.ylabel("error")
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'error.png'), dpi=300)
    plt.close()

    error_list = np.array(error_list)
    np.savetxt(os.path.join(save_dir, f'error.txt'), error_list)

    # vis
    # for cam in range(4):
    #     img_path = os.path.join(save_dir, f"camera_{cam}")
    #     frame_rate = 4
    #     height = 360
    #     width = 640
    #     pred_out_path = os.path.join(img_path, "pred.mp4")
    #     # ffmpeg -loglevel panic -r 4 -f image2 -s 640x360 -pattern_type glob -i '/mnt/nvme1n1p1/baoyu/vis/rollout-rope-model_100/rope/900/long/camera_0/*_pred.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p /mnt/nvme1n1p1/baoyu/vis/rollout-rope-model_100/rope/900/long/camera_0/pred.mp4 -y
    #     os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_pred.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {pred_out_path} -y")
    #     gt_out_path = os.path.join(img_path, "gt.mp4")
    #     os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_gt.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {gt_out_path} -y")
    #     both_out_path = os.path.join(img_path, "both.mp4")
    #     os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_both.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {both_out_path} -y")
    
    # vis
    for cam in range(4):
        img_path = os.path.join(save_dir, f"camera_{cam}")
        fps = 10
        pred_out_path = os.path.join(img_path, "pred.mp4")
        moviepy_merge_video(img_path, 'pred', pred_out_path, fps)
        gt_out_path = os.path.join(img_path, "gt.mp4")
        moviepy_merge_video(img_path, 'gt', gt_out_path, fps)
        both_out_path = os.path.join(img_path, "both.mp4")
        moviepy_merge_video(img_path, 'both', both_out_path, fps)

    return error_list


def rollout_episode_pushes(model, device, dataset, material_config, pairs, episode_idx, physics_param, save_dir, all_particles_pos, all_tool_states):
    n_his = model.model_config['n_his']

    # state_noise = 0.0
    # phys_noise = 0.0

    # load pushes
    steps = np.load(os.path.join(dataset['data_dir'], f"episode_{episode_idx}", "steps.npy"))

    error_list_pushes = []
    for i in range(len(steps)-1):
        # set valid pairs, which the first frame is the current step
        valid_pairs = pairs[pairs[:, 0] == steps[i]]
        try:
            assert len(valid_pairs) > 0
        except:
            import ipdb; ipdb.set_trace()
        pair = valid_pairs[0]

        start = pair[n_his-1]
        end = pair[n_his]

        graph, fps_idx_list = construct_graph(dataset, n_his, pair, episode_idx, physics_param, material_config, all_particles_pos, all_tool_states)
        
        error_list = rollout_from_start_graph(graph, model, material_config, device, dataset, episode_idx, start, end, 
                get_next_pair_or_break_episode_pushes, fps_idx_list, pairs, save_dir, all_particles_pos, all_tool_states)
        
        error_list_pushes.append(error_list)

        # plot error
        plt.figure(figsize=(10, 5))
        plt.plot(error_list)
        plt.xlabel("time step")
        plt.ylabel("error")
        plt.grid()
        plt.savefig(os.path.join(save_dir, f'error_{i}.png'), dpi=300)
        plt.close()

        error_list = np.array(error_list)
        np.savetxt(os.path.join(save_dir, f'error_{i}.txt'), error_list)

    # # vis
    # for cam in range(4):
    #     img_path = os.path.join(save_dir, f"camera_{cam}")
    #     frame_rate = 4
    #     height = 360
    #     width = 640
    #     pred_out_path = os.path.join(img_path, "pred.mp4")
    #     os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_pred.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {pred_out_path} -y")
    #     gt_out_path = os.path.join(img_path, "gt.mp4")
    #     os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_gt.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {gt_out_path} -y")
    #     both_out_path = os.path.join(img_path, "both.mp4")
    #     os.system(f"ffmpeg -loglevel panic -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_both.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {both_out_path} -y")

    # vis
    for cam in range(4):
        img_path = os.path.join(save_dir, f"camera_{cam}")
        fps = 10
        pred_out_path = os.path.join(img_path, "pred.mp4")
        moviepy_merge_video(img_path, 'pred', pred_out_path, fps)
        gt_out_path = os.path.join(img_path, "gt.mp4")
        moviepy_merge_video(img_path, 'gt', gt_out_path, fps)
        both_out_path = os.path.join(img_path, "both.mp4")
        moviepy_merge_video(img_path, 'both', both_out_path, fps)
        
    return error_list_pushes


def rollout_dataset(model, device, dataset, material_config, save_dir):
    pair_lists, physics_params = load_dataset(dataset, material_config, phase='valid')
    print(f'Loaded {len(pair_lists)} pairs from {dataset["name"]}')
    
    # load all particles and tool states
    data_dir = dataset['data_dir']
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    all_particles_pos = []
    all_tool_states = []
    for episode_idx in range(num_episodes):
        particles_pos = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
        tool_states = np.load(os.path.join(data_dir, f"episode_{episode_idx}/processed_eef_states.npy"))
        all_particles_pos.append(particles_pos)
        all_tool_states.append(tool_states)

    total_error_long = []
    total_error_short = []

    episode_idx_list = sorted(list(np.unique(pair_lists[:, 0]).astype(int)))
    for episode_idx in episode_idx_list:
        pair_lists_episode = pair_lists[pair_lists[:, 0] == episode_idx][:, 1:]
        physics_params_episode = physics_params[episode_idx]
        
        save_dir_episode = os.path.join(save_dir, f"{episode_idx}", "long")
        os.makedirs(save_dir_episode, exist_ok=True)
        error_list_long = rollout_episode(model, device, dataset, material_config, pair_lists_episode, episode_idx, 
                    physics_params_episode, save_dir_episode, all_particles_pos, all_tool_states)
        total_error_long.append(error_list_long)
        
        save_dir_episode_pushes = os.path.join(save_dir, f"{episode_idx}", "short")
        os.makedirs(save_dir_episode_pushes, exist_ok=True)
        error_list_short = rollout_episode_pushes(model, device, dataset, material_config, pair_lists_episode, episode_idx,
                    physics_params_episode, save_dir_episode_pushes, all_particles_pos, all_tool_states)
        total_error_short.extend(error_list_short)

    
    for (total_error, save_name) in zip([total_error_long, total_error_short], ['error_long', 'error_short']):
        
        max_step = max([len(total_error[i]) for i in range(len(total_error))])
        min_step = min([len(total_error[i]) for i in range(len(total_error))])
        step_error = np.zeros((min_step, len(total_error)))
        for i in range(min_step):
            for j in range(len(total_error)):
                # step_error[i] = np.mean([total_error[j][i] for j in range(len(total_error)) if i < len(total_error[j])])
                step_error[i, j] = total_error[j][i]

        # vis error
        # step_mean_error = step_error.mean(1)
        np.savetxt(os.path.join(save_dir, f'{save_name}.txt'), step_error)

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

        plt.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300)
        plt.close()


def rollout(config, epoch, out_dir_root):
    train_config = config['train_config']
    dataset_config = config['dataset_config']
    model_config = config['model_config']
    material_config = config['material_config']

    set_seed(train_config['random_seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = train_config['out_dir'].split('/')[-1]
    save_dir = os.path.join(out_dir_root, f"rollout-{run_name}-model_{epoch}")
    # save_dir = f"/mnt/sda/vis/rollout-{run_name}-model_{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(train_config['out_dir'], 'checkpoints', 'model_{}.pth'.format(epoch))

    model_config['n_his'] = train_config['n_his']
    model = DynamicsPredictor(model_config, material_config, device)
    model.to(device)

    mse_loss = torch.nn.MSELoss()
    loss_funcs = [(mse_loss, 1)]

    model.eval()
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))

    # vis
    # point_size = 5
    # line_size = 2
    # line_alpha = 0.5
    # colormap = rgb_colormap(repeat=100)  # only red

    assert len(dataset_config['datasets']) == 1, 'only support single dataset'

    for i, dataset in enumerate(dataset_config['datasets']):
        print(f'Rolling out on dataset {dataset["name"]} at {dataset["data_dir"]}')
        save_dir_dataset = os.path.join(save_dir, dataset['name'])
        os.makedirs(save_dir_dataset, exist_ok=True)
        rollout_dataset(model, device, dataset, material_config, save_dir_dataset)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/debug.yaml')
    arg_parser.add_argument('--epoch', type=str, default=100)
    args = arg_parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    out_dir_root = f"/mnt/nvme1n1p1/baoyu/vis"
    rollout(config, args.epoch, out_dir_root)