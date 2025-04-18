import os
import glob
import json
import numpy as np

import torch 
from torch.utils.data import Dataset

from dgl.geometry import farthest_point_sampler
from utils import pad, pad_torch, fps_rad_idx

def construct_edges_from_states(states, adj_thresh, mask, tool_mask, no_self_edge=False, pushing_direction=None):  # helper function for construct_graph
    '''
    # :param states: (B, N+2M, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N+2M) torch tensor, true when index is a valid particle
    # :param tool_mask: (B, N+2M) torch tensor, true when index is a valid tool particle
    # :param pushing_direction: (B, 3) torch tensor, pushing direction for each eef particle
    
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor
    '''
    no_self_edge = False

    B, N, state_dim = states.shape
    # print(f'states shape: {states.shape}') # (64, 300, 3)
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)
    # print(f's_receiv shape: {s_receiv.shape}; s_sender shape: {s_sender.shape}') # (64, 300, 300, 3)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    # convert threshold to tensor
    threshold = torch.tensor(threshold, device=states.device, dtype=states.dtype)  # (B, )
    
    s_diff = s_receiv - s_sender # (B, N, N, 3)
    dis = torch.sum(s_diff ** 2, -1)
    mask_1 = mask[:, :, None].repeat(1, 1, N)  # particle receiver
    mask_2 = mask[:, None, :].repeat(1, N, 1)  # particle sender
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)  # tool particle receiver
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)  # tool particle sender
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations
    
    # obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
    # obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender

    # obj_tool_mask = -1. * obj_tool_mask_1 + 1. * obj_tool_mask_2

    # pushing_direction = pushing_direction[:, None, None, :].repeat(1, N, N, 1)  # (B, N, N, 3)
    # pushing_diff = pushing_direction * obj_tool_mask[:, :, :, None]  # (B, N, N, 3)
    
    # # make pushing_diff and s_diff have the same device
    # pushing_diff = pushing_diff.to(device=states.device, dtype=states.dtype)
    # s_diff = s_diff.to(device=states.device, dtype=states.dtype)

    # pushing_mask = pushing_diff * s_diff  # (B, N, N, 3)
    # pushing_mask = torch.sum(pushing_mask, -1) < 0  # (B, N, N)
    # dis[pushing_mask] = 1e10  # avoid tool to obj relations in reverse pushing direction

    adj_matrix = ((dis - threshold[:, None, None]) < 0).to(torch.float32) # (B, N, N)
    # print(f'adj_matrix shape: {adj_matrix.shape}') # (64, 300, 300)
    # print(adj_matrix) # 0 or 1
    # adj_matrix = adj_matrix.to(device=states.device, dtype=states.dtype)

    # remove self edge
    if no_self_edge:
        self_edge_mask = torch.eye(N, device=states.device, dtype=states.dtype)[None, :, :]
        adj_matrix = adj_matrix * (1 - self_edge_mask)

    # add topk constraints
    # Check before train and rollout
    topk = 10 #TODO: hyperparameter
    if topk > dis.shape[-1]:
        topk = dis.shape[-1]
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix
    # print(f'adj_matrix shape: {adj_matrix.shape}') # (64, 300, 300)
    
    n_rels = adj_matrix.sum(dim=(1,2))
    # print(f'n_rels: {n_rels}') # (64)
    # print(n_rels.shape, (mask * 1.0).sum(-1).mean().item(), n_rels.mean().item())
    n_rel = n_rels.max().long().item()
    # print('relation num:', n_rel)
    
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    # print(f'Rr shape: {Rr.shape}; Rs shape: {Rs.shape}') # (64, 410, 300); (64, 410, 300)
    # print(Rr, Rs)
    
    return Rr, Rs

def load_pairs(pairs_path, episode_range):
    pair_lists = []
    for episode_idx in episode_range:
        n_pushes = len(list(glob.glob(os.path.join(pairs_path, f'{episode_idx}_*.txt'))))
        for push_idx in range(n_pushes):
            frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}_{push_idx}.txt'))
            if len(frame_pairs.shape) == 1: continue
            episodes = np.ones((frame_pairs.shape[0], 1)) * episode_idx
            pairs = np.concatenate([episodes, frame_pairs], axis=1)
            pair_lists.extend(pairs)
    pair_lists = np.array(pair_lists).astype(int)
    return pair_lists

def load_dataset(dataset, phase='train'):
    data_dir = dataset["data_dir"]
    prep_data_dir = dataset["prep_data_dir"]

    # load kypts paths
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Found num_episodes: {num_episodes}")

    # load data pairs
    episode_range_phase = range(
        int(num_episodes * dataset["ratio"][phase][0]),
        int(num_episodes * dataset["ratio"][phase][1]),
    )
    pairs_path = os.path.join(prep_data_dir, 'frame_pairs')
    pair_lists = load_pairs(pairs_path, episode_range_phase)
    print(f'{phase} dataset has {len(list(episode_range_phase))} episodes, {len(pair_lists)} frame pairs')

    # physics_params = []
    # for episode_idx in range(num_episodes):
    #     physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property_params.json")
    #     with open(physics_path) as f:
    #         properties = json.load(f)
        
    #     physics_params_episode = {}

    #     for material_name in dataset["materials"]:
    #         material_params = material_config[material_name]['physics_params']

    #         used_params = []
    #         for item in material_params:
    #             if item['name'] in properties.keys() and item['use']:
    #                 range_min = item['min']
    #                 range_max = item['max']
    #                 used_params.append((properties[item['name']] - range_min) / (range_max - range_min + 1e-6))
            
    #         used_params = np.array(used_params).astype(np.float32)
    #         # used_params = used_params * 2. - 1. #TODO: normalize to [-1, 1]
    #         physics_params_episode[material_name] = used_params
    #     physics_params.append(physics_params_episode)

    return pair_lists

class DynDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        phase='train',
    ):
        assert phase in ['train', 'valid']
        print(f'Loading {phase} dataset...')
        self.phase = phase
        self.dataset_config = dataset_config

        self.pair_lists = load_dataset(dataset_config, phase)

        num_episodes = len(list(glob.glob(os.path.join(dataset_config["data_dir"], f"episode_*"))))
        data_dir = dataset_config["data_dir"]

        # save all particles and tool states
        self.all_particle_pos = []
        self.all_tool_states = []
        for episode_idx in range(num_episodes):
            particles_pos = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
            tool_states = np.load(os.path.join(data_dir, f"episode_{episode_idx}/processed_eef_states.npy"))
            # print(f'episode {episode_idx}: particles_pos: {particles_pos.shape}, tool_states shape: {tool_states.shape}.')
            assert particles_pos.shape[0] == tool_states.shape[0], 'particle pos and tool states should have the same length'

            self.all_particle_pos.append(particles_pos) 
            self.all_tool_states.append(tool_states)

    def __len__(self):
        return len(self.pair_lists)

    def __getitem__(self, i):
        episode_idx = self.pair_lists[i][0].astype(int)
        pair = self.pair_lists[i][1:].astype(int)

        n_his = self.dataset_config['n_his']
        n_future = self.dataset_config['n_future']

        max_nobj = self.dataset_config['max_nobj']
        num_obj = self.dataset_config['max_n']
        max_ntool = self.dataset_config['max_ntool']
        num_tool = self.dataset_config['max_tool']
        max_nR = self.dataset_config['max_nR']
        fps_radius_range = self.dataset_config['fps_radius_range']
        adj_radius_range = self.dataset_config['adj_radius_range']
        state_noise = self.dataset_config['state_noise'][self.phase]

        # get history keypoints
        obj_kps, tool_kps = [], []
        for i in range(len(pair)):
            frame_idx = pair[i]
            # obj_kp: (1, num_obj_points, 3)
            # tool_kp: (num_tool_points, 3)
            # obj_ptcls = self.all_particle_pos[episode_idx]
            # obj_kp, tool_kp = extract_kp_single_frame(dataset_config['data_dir'], episode_idx, frame_idx)
            # print(obj_kp.shape, tool_kp.shape) 
            
            # TODO
            # if num_obj == 1:
            #     obj_kp = self.all_particle_pos[episode_idx][frame_idx][None] # (1, num_obj_points, 3)
            # else:
            #     obj_kp = self.all_particle_pos[episode_idx][frame_idx] # (num_obj, num_obj_points, 3)
            obj_kp = self.all_particle_pos[episode_idx][frame_idx]
            tool_kp = self.all_tool_states[episode_idx][frame_idx] # (num_tool_points, 3)
            
            # print(obj_kp.shape, tool_kp.shape)
            
            obj_kps.append(obj_kp)
            tool_kps.append(tool_kp) # (7, num_tool_points, 3)
        
        obj_kp_start = obj_kps[n_his - 1]
        state_dim = obj_kp_start.shape[-1]
        instance_num = len(obj_kp_start)
        # assert instance_num == 1, 'only support single object'
        
        fps_idx_list = []
        
        # old sampling using raw particles
        # TODO: change it to preprocessing?
        for j in range(len(obj_kp_start)):
            # farthers point sampling
            particle_tensor = torch.from_numpy(obj_kp_start[j]).float()[None, ...] # convert the first dim to None
            fps_idx_tensor = farthest_point_sampler(particle_tensor, min(max_nobj, particle_tensor.shape[1]), start_idx=np.random.randint(0, obj_kp_start[j].shape[0]))[0]
            fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)
            
            # downsample to uniform radius
            downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
            fps_radius = np.random.uniform(fps_radius_range[0], fps_radius_range[1])
            _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
            fps_idx_2 = fps_idx_2.astype(np.int32)
            # print('fps_points:', fps_idx_2.shape)
            fps_idx = fps_idx_1[fps_idx_2]
            fps_idx_list.append(fps_idx)

        # downsample to get current obj kp
        obj_kp_start = [obj_kp_start[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
        obj_kp_start = np.concatenate(obj_kp_start, axis=0) # (N, 3)
        obj_kp_num = obj_kp_start.shape[0]
        
        # get current state delta 
        # tool_kp: (2, num_tool_points, 3)
        tool_kp = np.stack(tool_kps[n_his-1:n_his+1], axis=0)
        tool_kp_num = tool_kp.shape[1]
        # print(tool_kp.shape)

        ## states (object + tool, 3)
        states_delta = np.zeros((max_nobj + max_ntool * num_tool, state_dim), dtype=np.float32)
        states_delta[max_nobj : max_nobj + tool_kp_num] = tool_kp[1] - tool_kp[0]

        # new: get pushing direction
        pushing_direction = states_delta[max_nobj]  # (3, )
        
        # load future states
        obj_kp_future = np.zeros((n_future, max_nobj, state_dim), dtype=np.float32)
        # obj_future_mask = np.ones(n_future).astype(bool) # (n_future, )
        for fi in range(n_future):
            obj_kp_fu = obj_kps[n_his + fi]
            obj_kp_fu = [obj_kp_fu[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_fu = np.concatenate(obj_kp_fu, axis=0) 
            obj_kp_fu = pad(obj_kp_fu, max_nobj)
            obj_kp_future[fi] = obj_kp_fu
        
        # load future tool keypoints
        tool_future = np.zeros((n_future - 1, max_nobj + max_ntool * num_tool, state_dim), dtype=np.float32)
        states_delta_future = np.zeros((n_future - 1, max_nobj + max_ntool * num_tool, state_dim), dtype=np.float32)
        for fi in range(n_future - 1):
            # dynamic tool
            tool_kp_future = tool_kps[n_his+fi:n_his+fi+2]
            tool_kp_future = np.stack(tool_kp_future, axis=0) # (2, max_ntool, 3)
            tool_future[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_future[0]
            states_delta_future[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_future[1] - tool_kp_future[0]
        
        # load history states
        state_history = np.zeros((n_his, max_nobj + max_ntool * num_tool, state_dim), dtype=np.float32)
        for fi in range(n_his):
            # object 
            obj_kp_his = obj_kps[fi]
            obj_kp_his = [obj_kp_his[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_his = np.concatenate(obj_kp_his, axis=0)
            obj_kp_his = pad(obj_kp_his, max_nobj)
            state_history[fi, :max_nobj] = obj_kp_his
            
            # dynamic tool
            tool_kp_his = tool_kps[fi]
            tool_kp_his = pad(tool_kp_his, max_ntool)
            state_history[fi, max_nobj : max_nobj + max_ntool] = tool_kp_his
        
        # load masks
        state_mask = np.zeros((max_nobj + max_ntool * num_tool), dtype=bool)
        state_mask[:obj_kp_num] = True # obj
        state_mask[max_nobj : max_nobj + tool_kp_num] = True # dynamic tool
        
        obj_mask = np.zeros((max_nobj,), dtype=bool)
        obj_mask[:obj_kp_num] = True
        
        tool_mask = np.zeros((max_nobj + max_ntool * num_tool,), dtype=bool)
        tool_mask[max_nobj : max_nobj + tool_kp_num] = True # dynamic tool
        
        # construct instance information
        p_rigid = np.zeros(num_obj, dtype=np.float32)
        p_instance = np.zeros((max_nobj, num_obj), dtype=np.float32)
        # j_perm = np.random.permutation(instance_num)
        # ptcl_cnt = 0
        # # sanity check
        # assert sum([len(fps_idx_list[j]) for j in range(len(fps_idx_list))]) == obj_kp_num
        # # fill in p_instance
        # for j in range(instance_num):
        #     p_instance[ptcl_cnt:ptcl_cnt + len(fps_idx_list[j_perm[j]]), j_perm[j]] = 1
        #     ptcl_cnt += len(fps_idx_list[j_perm[j]])
        ptcl_cnt = 0
        for j in range(instance_num):
            p_instance[ptcl_cnt:ptcl_cnt + len(fps_idx_list[j]), j] = 1
            ptcl_cnt += len(fps_idx_list[j])

        # construct physics information
        # physics_param = self.physics_params[dataset_idx][episode_idx]  # dict
        # for material_name in dataset_config['materials']:
        #     if material_name not in physics_param.keys():
        #         raise ValueError(f'Physics parameter {material_name} not found in {dataset_config["data_dir"]}')
        #     physics_param[material_name] += np.random.uniform(-phys_noise, phys_noise, 
        #             size=physics_param[material_name].shape)
        
        # new: construct physics information for each particle
        # material_idx = np.zeros((max_nobj, len(self.material_config['material_index'])), dtype=np.int32)
        # assert len(dataset_config['materials']) == 1, 'only support single material'
        # material_idx[:obj_kp_num, self.material_config['material_index'][dataset_config['materials'][0]]] = 1
        
        # construct attributes
        attr_dim = 1 + num_tool # (obj, tool)
        attrs = np.zeros((max_nobj + max_ntool * num_tool, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[max_nobj : max_nobj + tool_kp_num, 1] = 1.
        
        # state randomness
        state_history += np.random.uniform(-state_noise, state_noise, size=state_history.shape)
        
        # TODO: rotation randomness
        random_rot = np.random.uniform(-np.pi, np.pi)
        if state_dim == 3:
            rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                                [np.sin(random_rot), np.cos(random_rot), 0],
                            [0, 0, 1]], dtype=state_history.dtype) # 2D rotation matrix in xy plane
        elif state_dim == 2:
            rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot)],
                                [np.sin(random_rot), np.cos(random_rot)]], dtype=state_history.dtype)
        state_history = state_history @ rot_mat[None]
        states_delta = states_delta @ rot_mat
        tool_future = tool_future @ rot_mat[None]
        states_delta_future = states_delta_future @ rot_mat[None]
        obj_kp_future = obj_kp_future @ rot_mat[None]
        
        # translation randomness
        # random_translation = np.random.uniform(-1, 1, size=3)
        # state_history += random_translation[None, None]
        # states_delta += random_translation[None]
        # tool_future += random_translation[None, None]
        # states_delta_future += random_translation[None, None]
        # obj_kp_future += random_translation[None, None]
        
        # numpy to torch
        # state_history = torch.from_numpy(state_history).float()
        # states_delta = torch.from_numpy(states_delta).float()
        # tool_future = torch.from_numpy(tool_future).float()
        # states_delta_future = torch.from_numpy(states_delta_future).float()
        # obj_kp_future = torch.from_numpy(obj_kp_future).float()
        # # obj_future_mask = torch.from_numpy(obj_future_mask)
        # attrs = torch.from_numpy(attrs).float()
        # p_rigid = torch.from_numpy(p_rigid).float()
        # p_instance = torch.from_numpy(p_instance).float()
        # physics_param = {k: torch.from_numpy(v).float() for k, v in physics_param.items()}
        # material_idx = torch.from_numpy(material_idx).long()
        # state_mask = torch.from_numpy(state_mask)
        # tool_mask = torch.from_numpy(tool_mask)
        # obj_mask = torch.from_numpy(obj_mask)

        # construct edges
        # adj_thresh = np.random.uniform(*adj_radius_range)
        # adj_thresh = torch.tensor([adj_thresh], device=state_history.device, dtype=state_history.dtype)
        # Rr, Rs = construct_edges_from_states(state_history[-1][None], adj_thresh, 
        #             mask=state_mask[None], tool_mask=tool_mask[None], no_self_edge=True)
        # Rr = Rr[0]
        # Rs = Rs[0]
        # Rr = pad_torch(Rr, max_nR)
        # Rs = pad_torch(Rs, max_nR)

        # save graph
        graph = {
            ## N: max_nobj, M: max_ntool
            # input info 
            "state": state_history, # (n_his, N+2M, state_dim)
            "action": states_delta, # (N+2M, state_dim)
            
            # future info
            "tool_future": tool_future, # (n_future-1, N+2M, state_dim)
            "action_future": states_delta_future, # (n_future-1, N+2M, state_dim)
            
            # "Rr": Rr, # (n_rel, N)
            # "Rs": Rs, # (n_rel, N)
            
            # gt info
            "state_future": obj_kp_future, # (n_future, N, state_dim)
            # "state_future_mask": obj_future_mask, # (n_future, )
            
            # attr info
            "attrs": attrs, # (N+2M, attr_dim)
            "p_rigid": p_rigid, # (n_instance, )
            "p_instance": p_instance, # (N, n_instance)
            # "physics_param": physics_param, # (N, phys_dim)
            "state_mask": state_mask, # (N+2M, )
            "tool_mask": tool_mask, # (N+2M, )
            "obj_mask": obj_mask, # (N, )

            "pushing_direction": pushing_direction, # (3, )
        }

        # for material_name, material_dim in self.materials.items():
        #     if material_name in physics_param.keys():
        #         graph[material_name + '_physics_param'] = physics_param[material_name]
        #     else:
        #         graph[material_name + '_physics_param'] = torch.zeros(material_dim, dtype=torch.float32)

        return graph
