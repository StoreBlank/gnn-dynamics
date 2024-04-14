import os
import glob
import json
import numpy as np

import torch 
from torch.utils.data import Dataset

from dgl.geometry import farthest_point_sampler
from utils import pad, pad_torch, fps_rad_idx

def construct_edges_from_states(states, adj_thresh, mask, tool_mask, material_types, no_self_edge=False):  # helper function for construct_graph
    '''
    # :param states: (B, N+2M, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N+2M) torch tensor, true when index is a valid particle
    # :param tool_mask: (B, N+2M) torch tensor, true when index is a valid tool particle
    # :param material_type: (B, ), 'cloth' or 'rope' or 'granular'
    # :param no_self_edge: bool, whether to remove self edge
    
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

    adj_matrix = ((dis - threshold[:, None, None]) < 0).to(torch.float32) # (B, N, N)
    # print(f'adj_matrix shape: {adj_matrix.shape}') # (64, 300, 300)
    # print(adj_matrix) # 0 or 1
    # adj_matrix = adj_matrix.to(device=states.device, dtype=states.dtype)

    # remove self edge
    if no_self_edge:
        self_edge_mask = torch.eye(N, device=states.device, dtype=states.dtype)[None, :, :]
        adj_matrix = adj_matrix * (1 - self_edge_mask)

    # add topk constraints
    topk = 10 #TODO: hyperparameter
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix
    # print(f'adj_matrix shape: {adj_matrix.shape}') # (B, 300, 300)
    
    # for cloth, connect eef to all object particles due to slow message passing
    obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender
    
    material_mask = torch.tensor([material_type == 'cloth' for material_type in material_types], device=states.device, dtype=bool) # (B, )
    batch_mask = torch.ones((B, N, N), device=states.device, dtype=bool) # (B, N, N)
    batch_mask = batch_mask * material_mask[:, None, None] # (B, N, N)
    
    batch_obj_tool_mask_1 = obj_tool_mask_1 * batch_mask  # (B, N, N)
    neg_batch_obj_tool_mask_1 = obj_tool_mask_1 * (~batch_mask)  # (B, N, N)
    batch_obj_tool_mask_2 = obj_tool_mask_2 * batch_mask  # (B, N, N)
    neg_batch_obj_tool_mask_2 = obj_tool_mask_2 * (~batch_mask)  # (B, N, N)
    
    adj_matrix[batch_obj_tool_mask_1] = 0
    adj_matrix[batch_obj_tool_mask_2] = 1
    adj_matrix[neg_batch_obj_tool_mask_1] = 0
    adj_matrix[neg_batch_obj_tool_mask_2] = 0
    adj_matrix[tool_mask_12] = 0  # avoid tool to tool relations
    
    # import pdb; pdb.set_trace()
        
    # construct relation graph
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

def load_dataset(dataset_config, material_config, phase='train'):
    """
    :param dataset_config: dict, dataset configuration
    :param material_config: dict, material configuration
    :param phase: str, 'train' or 'valid'
    
    :return: pair_lists: np.array, (num_pairs, 2), episode_idx, frame_idx
    :return: material_types: list, one-hot encoding
    :return: physics_params: list, physics parameters
    """
    
    data_dir = dataset_config['data_dir']
    prep_data_dir = dataset_config['prep_data_dir']
    dataset = dataset_config['datasets']

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

    physics_params = []
    material_types = [] 
    for episode_idx in range(num_episodes):
        # load material type
        with open(os.path.join(data_dir, f"episode_{episode_idx}/material.txt"), "r") as f:
            material_name = f.read().strip()
        material_types.append(material_name)
        
        # load physics parameters
        physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property_params.json")
        with open(physics_path) as f:
            properties = json.load(f)
        
        material_params = material_config[material_name]['physics_params']
        used_params = []
        for item in material_params:
            if item['name'] in properties.keys() and item['use']:
                range_min = item['min']
                range_max = item['max']
                used_params.append((properties[item['name']] - range_min) / (range_max - range_min + 1e-6)) # normalize to [0, 1]
        
        used_params = np.array(used_params).astype(np.float32)
        physics_params.append(used_params)

    return pair_lists, material_types, physics_params  

class DynDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        material_config,
        phase='train',
    ):
        assert phase in ['train', 'valid']
        print(f'Loading {phase} dataset...')
        self.phase = phase

        self.dataset_config = dataset_config
        self.material_config = material_config
        
        data_dir = dataset_config['data_dir']
        
        # generate pair lists, material types, physics params
        print(f'Setting up dataset mixed at {data_dir}')
        pair_lists, material_types, physics_params = load_dataset(dataset_config, material_config, phase)
        pair_lists = np.concatenate([np.ones((pair_lists.shape[0], 1)) * 0, pair_lists], axis=1)
        print('Length of dataset is', len(pair_lists))
        
        self.pair_lists = np.array(pair_lists) # (num_pairs, 2), episode_idx, frame_idx
        self.material_types = material_types # size: num_episodes
        self.physics_params = physics_params # size: num_episodes
        
        # save all particles and tool states
        num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*")))) 
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

        episode_idx = self.pair_lists[i][1].astype(int)
        pair = self.pair_lists[i][2:].astype(int)
        material_type = self.material_types[episode_idx]
        
        n_his = self.dataset_config['n_his']
        n_future = self.dataset_config['n_future']

        dataset_config = self.dataset_config['datasets']['material'][material_type]
        max_n = dataset_config['max_n']
        max_tool = dataset_config['max_tool']
        max_nobj = dataset_config['max_nobj']
        max_ntool = dataset_config['max_ntool']
        max_nR = dataset_config['max_nR']
        fps_radius_range = dataset_config['fps_radius_range']
        adj_radius_range = dataset_config['adj_radius_range']
        state_noise = dataset_config['state_noise'][self.phase]
        phys_noise = dataset_config['phys_noise'][self.phase]
        
        material_category = self.material_config['material_index']
        material_index = material_category[material_type]
        
        # get history keypoints
        obj_kps, tool_kps = [], []
        for i in range(len(pair)):
            frame_idx = pair[i]
            
            # extract keypoints
            obj_kp = self.all_particle_pos[episode_idx][frame_idx][None] # (1, num_obj_points, 3)
            tool_kp = self.all_tool_states[episode_idx][frame_idx] # (num_tool_points, 3)
            # print(obj_kp.shape, tool_kp.shape)
            
            obj_kps.append(obj_kp)
            tool_kps.append(tool_kp) # (7, num_tool_points, 3)
        
        obj_kp_start = obj_kps[n_his - 1]
        instance_num = len(obj_kp_start)
        assert instance_num == 1, 'only support single object'
        
        fps_idx_list = []
        
        # old sampling using raw particles
        for j in range(len(obj_kp_start)):
            # farthers point sampling
            particle_tensor = torch.from_numpy(obj_kp_start[j]).float()[None, ...] # convert the first dim to None
            fps_idx_tensor = farthest_point_sampler(particle_tensor, max_nobj, start_idx=np.random.randint(0, obj_kp_start[j].shape[0]))[0]
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
        states_delta = np.zeros((max_nobj + max_ntool * max_tool, 3), dtype=np.float32)
        states_delta[max_nobj : max_nobj + tool_kp_num] = tool_kp[1] - tool_kp[0]

        # new: get pushing direction
        pushing_direction = states_delta[max_nobj]  # (3, )
        
        # load future states
        obj_kp_future = np.zeros((n_future, max_nobj, 3), dtype=np.float32)
        # obj_future_mask = np.ones(n_future).astype(bool) # (n_future, )
        for fi in range(n_future):
            obj_kp_fu = obj_kps[n_his + fi]
            obj_kp_fu = [obj_kp_fu[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_fu = np.concatenate(obj_kp_fu, axis=0) 
            obj_kp_fu = pad(obj_kp_fu, max_nobj)
            obj_kp_future[fi] = obj_kp_fu
        
        # load future tool keypoints
        tool_future = np.zeros((n_future - 1, max_nobj + max_ntool * max_tool, 3), dtype=np.float32)
        states_delta_future = np.zeros((n_future - 1, max_nobj + max_ntool * max_tool, 3), dtype=np.float32)
        for fi in range(n_future - 1):
            # dynamic tool
            tool_kp_future = tool_kps[n_his+fi:n_his+fi+2]
            tool_kp_future = np.stack(tool_kp_future, axis=0) # (2, max_ntool, 3)
            tool_future[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_future[0]
            states_delta_future[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_future[1] - tool_kp_future[0]
        
        # load history states
        state_history = np.zeros((n_his, max_nobj + max_ntool * max_tool, 3), dtype=np.float32)
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
        state_mask = np.zeros((max_nobj + max_ntool * max_tool), dtype=bool)
        state_mask[:obj_kp_num] = True # obj
        state_mask[max_nobj : max_nobj + tool_kp_num] = True # dynamic tool
        
        obj_mask = np.zeros((max_nobj,), dtype=bool)
        obj_mask[:obj_kp_num] = True
        
        tool_mask = np.zeros((max_nobj + max_ntool * max_tool,), dtype=bool)
        tool_mask[max_nobj : max_nobj + tool_kp_num] = True # dynamic tool
        
        # construct instance information
        p_rigid = np.zeros(max_n, dtype=np.float32)
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
        physics_param = self.physics_params[episode_idx]  
        physics_param += np.random.uniform(-phys_noise, phys_noise, size=physics_param.shape)
        
        # construct material one-hot encoding
        material_idx = material_index
        material_encoding = np.zeros(len(material_category), dtype=np.int32)
        material_encoding[material_idx] = 1
        
        # construct adj_thresh_range based on material type
        # adj_radius_range = dataset_config['adj_radius_range']
        
        # construct attributes
        attr_dim = 2 # (obj, tool)
        attrs = np.zeros((max_nobj + max_ntool * max_tool, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[max_nobj : max_nobj + tool_kp_num, 1] = 1.
        
        # state randomness
        state_history += np.random.uniform(-state_noise, state_noise, size=state_history.shape)
        
        # rotation randomness
        random_rot = np.random.uniform(-np.pi, np.pi)
        rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                            [np.sin(random_rot), np.cos(random_rot), 0],
                           [0, 0, 1]], dtype=state_history.dtype) # 2D rotation matrix in xy plane
        state_history = state_history @ rot_mat[None]
        states_delta = states_delta @ rot_mat
        tool_future = tool_future @ rot_mat[None]
        states_delta_future = states_delta_future @ rot_mat[None]
        obj_kp_future = obj_kp_future @ rot_mat[None]

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
            "state_mask": state_mask, # (N+2M, )
            "tool_mask": tool_mask, # (N+2M, )
            "obj_mask": obj_mask, # (N, )

            "pushing_direction": pushing_direction, # (3, )
            
            "material_encoding": material_encoding, # (num_materials, )
            "physics_param": physics_param, # (physics_dim, )
        }

        return graph



