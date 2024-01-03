import os
import glob
import json
import numpy as np

import torch 
from torch.utils.data import Dataset

from dgl.geometry import farthest_point_sampler
from utils import pad, extract_kp_single_frame, fps_rad_idx

def construct_edges_from_states(states, adj_thresh, mask, tool_mask, no_self_edge=False):  # helper function for construct_graph
    '''
    # :param states: (B, N+2M, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N+2M) torch tensor, true when index is a valid particle
    # :param eef_mask: (B, N+2M) torch tensor, true when index is a valid tool particle
    
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor
    '''
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
    threshold = torch.tensor(threshold, device=states.device, dtype=states.dtype)
    
    dis = torch.sum((s_sender - s_receiv)**2, -1)
    mask_1 = mask[:, :, None].repeat(1, 1, N)
    mask_2 = mask[:, None, :].repeat(1, N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations
    
    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()
    # print(f'adj_matrix shape: {adj_matrix.shape}') # (64, 300, 300)
    # print(adj_matrix)
    # adj_matrix = adj_matrix.to(device=states.device, dtype=states.dtype)

    # remove self edge
    if no_self_edge:
        self_edge_mask = torch.eye(N, device=states.device, dtype=states.dtype)[None, :, :]
        adj_matrix = adj_matrix * (1 - self_edge_mask)

    # add topk constraints
    topk = 20
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix
    # print(f'adj_matrix shape: {adj_matrix.shape}') # (64, 300, 300)
    
    n_rels = adj_matrix.sum(dim=(1,2))
    # print(f'n_rels: {n_rels}') # (64)
    # print(n_rels.shape, (mask * 1.0).sum(-1).mean().item(), n_rels.mean().item())
    n_rel = n_rels.max().long().item()
    # print(f'n_rel: {n_rel}') # 410
    
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

class GranularToolDynDataset(Dataset):
    def __init__(
        self,
        data_dir, 
        prep_save_dir, 
        phase, 
        ratios, 
        dist_thresh,
        fps_radius_range, # radius for fps sampling
        adj_thresh_range, # threshold for constructing edges (not used here)
        n_future,
        n_his,
        max_n, # max number of objects
        max_nobj, # max number of object points
        max_ntool, # max number of points per tool TODO: now it is the same as the number of points per tool
        max_nR, # max number of relations (not used here)
        phys_noise = 0.0,
        canonical = False,
    ):
        self.phase = phase
        self.data_dir = data_dir
        print(f'Setting up GranularToolDynDataset')
        print(f'Setting up {phase} dataset, data_dir: {len(data_dir)}')
        
        self.n_his = n_his
        self.n_future = n_future
        self.dist_thresh = dist_thresh
        
        self.max_n = max_n
        self.max_nobj = max_nobj
        self.max_ntool = max_ntool
        self.fps_radius_range = fps_radius_range # for object points
        self.phys_noise = phys_noise
        
        self.obj_kypts_paths = []
        self.physics_paths = []
        
        self.static_tool_kypts_paths = []
        self.dynamic_tool_kypts_paths = []
        
        # load kypts paths
        num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
        print(f"Found num_episodes: {num_episodes}")
        frame_count = 0
        for episode_idx in range(num_episodes):
            n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
            obj_kypts_path = os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy")
            physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property_params.json")
            
            static_tool_kypts_path = os.path.join(data_dir, f"episode_{episode_idx}/dustpan_points.npy")
            dynamic_tool_kypts_path = os.path.join(data_dir, f"episode_{episode_idx}/sponge_points.npy")
            
            self.obj_kypts_paths.append(obj_kypts_path)
            self.physics_paths.append(physics_path)
            
            self.static_tool_kypts_paths.append(static_tool_kypts_path)
            self.dynamic_tool_kypts_paths.append(dynamic_tool_kypts_path)
            
            frame_count += n_frames
        
        print(f'Found {frame_count} frames in {data_dir}')
        
        # load data pairs
        save_dir = os.path.join(prep_save_dir, data_dir.split('/')[-1])
        pairs_path = os.path.join(save_dir, 'frame_pairs')
        self.pair_lists = []
        for episode_idx in range(num_episodes):
            n_pushes = len(list(glob.glob(os.path.join(pairs_path, f'{episode_idx}_*.txt'))))
            for push_idx in range(n_pushes):   
                frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}_{push_idx}.txt'))
                if len(frame_pairs.shape) == 1: 
                    continue
                episodes = np.ones((frame_pairs.shape[0], 1)) * episode_idx
                pairs = np.concatenate([episodes, frame_pairs], axis=1)
                self.pair_lists.extend(pairs)
        self.pair_lists = np.array(self.pair_lists)
        print('pair lists shape: ', self.pair_lists.shape)
        print(f'Found {len(self.pair_lists)} frame pairs in {pairs_path}')
        
        # load phys_params 
        physics_range = np.loadtxt(os.path.join(save_dir, 'phys_range.txt')).astype(np.float32)
        # physics_range = physics_range[:, 2:4]
        self.physics_params = []
        for episode_idx in range(num_episodes):
            physics_path = self.physics_paths[episode_idx]
            assert os.path.join(self.data_dir, f"episode_{episode_idx}/property_params.json") == physics_path
            with open(physics_path, "r") as f:
                properties = json.load(f)
            
            physics_param = np.array([
                # properties['particle_radius'],
                # properties['num_particles'],
                properties['granular_scale'],
                # properties['num_granular'],
                # properties['distribution_r'],
                properties['dynamic_friction'],
                properties['granular_mass']
            ]).astype(np.float32)
            
            physics_param = (physics_param - physics_range[0]) / (physics_range[1] - physics_range[0] + 1e-6)  # normalize
            self.physics_params.append(physics_param)
        self.physics_params = np.stack(self.physics_params, axis=0) # (N, phys_dim)
        
        # take ratio to split pari lists
        self.ratio = ratios
        print(f'Taking ratio {self.ratio} of {len(self.pair_lists)} keypoint files')
        self.pair_lists = self.pair_lists[int(len(self.pair_lists) * self.ratio[0]):int(len(self.pair_lists) * self.ratio[1])]
        print(f'{phase} dataset has {len(self.pair_lists)} pairs')
        
        episode_cnt = 0
        for episode_idx in range(num_episodes):
            episode_len = len(self.pair_lists[self.pair_lists[:, 0] == episode_idx])
            if episode_len != 0:
                episode_cnt += 1
        print(f'{phase} dataset has {episode_cnt} episodes')         
        
    def __len__(self):
        return len(self.pair_lists)
    
    def __getitem__(self, i):
        max_n = self.max_n
        max_nobj = self.max_nobj
        max_ntool = self.max_ntool
        
        n_his = self.n_his
        n_future = self.n_future
        fps_radius_range = self.fps_radius_range
        
        episode_idx = self.pair_lists[i, 0].astype(np.int32)
        pair = self.pair_lists[i, 1:].astype(np.int32)
        
        # get history keypoints
        obj_kps, static_tool_kps, dynamic_tool_kps = [], [], []
        for i in range(len(pair)):
            frame_idx = pair[i]
            # obj_kp: (1, num_obj_points, 3)
            # static_tool_kp: (num_static_tool_points, 3)
            # dynamic_tool_kp: (num_dynamic_tool_points, 3)
            obj_kp, static_tool_kp, dynamic_tool_kp = extract_kp_single_frame(self.data_dir, episode_idx, frame_idx)
            # print(obj_kp.shape, static_tool_kp.shape, dynamic_tool_kp.shape) 
            
            obj_kps.append(obj_kp)
            static_tool_kps.append(static_tool_kp)
            dynamic_tool_kps.append(dynamic_tool_kp)
        
        obj_kp_start = obj_kps[n_his - 1]
        instance_num = len(obj_kp_start)
        assert instance_num == 1, 'only support single object'
        
        self.fps_idx_list = []
        
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
            fps_idx = fps_idx_1[fps_idx_2]
            self.fps_idx_list.append(fps_idx)
        
        # downsample to get current obj kp
        obj_kp_start = [obj_kp_start[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
        obj_kp_start = np.concatenate(obj_kp_start, axis=0) # (N, 3)
        obj_kp_num = obj_kp_start.shape[0]
        
        # get current state delta
        static_tool_kp = np.stack(static_tool_kps[n_his-1:n_his+1], axis=0) 
        dynamic_tool_kp = np.stack(dynamic_tool_kps[n_his-1:n_his+1], axis=0)
        static_tool_kp_num, dynamic_tool_kp_num = static_tool_kp.shape[1], dynamic_tool_kp.shape[1]
        # print(static_tool_kp.shape, dynamic_tool_kp.shape) # (2, 100, 3)
        
        ## states (object + dynamic_tool + static_tool, 3)
        states_delta = np.zeros((max_nobj + max_ntool * 2, 3), dtype=np.float32)
        states_delta[max_nobj : max_nobj + dynamic_tool_kp_num] = dynamic_tool_kp[1] - dynamic_tool_kp[0]
        assert (static_tool_kp[1] == static_tool_kp[0]).all(), 'static tool should be static'
        
        # load future states
        obj_kp_future = np.zeros((n_future, max_nobj, 3), dtype=np.float32)
        obj_future_mask = np.ones(n_future).astype(bool) # (n_future, )
        for fi in range(n_future):
            obj_kp_fu = obj_kps[n_his + fi]
            obj_kp_fu = [obj_kp_fu[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
            obj_kp_fu = np.concatenate(obj_kp_fu, axis=0) 
            obj_kp_fu = pad(obj_kp_fu, max_nobj)
            obj_kp_future[fi] = obj_kp_fu
        
        # load future tool keypoints
        tool_future = np.zeros((n_future - 1, max_nobj + max_ntool * 2, 3), dtype=np.float32)
        states_delta_future = np.zeros((n_future - 1, max_nobj + max_ntool * 2, 3), dtype=np.float32)
        for fi in range(n_future - 1):
            # dynamic tool
            dynamic_tool_kp_future = dynamic_tool_kps[n_his+fi:n_his+fi+2]
            dynamic_tool_kp_future = np.stack(dynamic_tool_kp_future, axis=0) # (2, max_ntool, 3)
            tool_future[fi, max_nobj : max_nobj + dynamic_tool_kp_num] = dynamic_tool_kp_future[0]
            states_delta_future[fi, max_nobj : max_nobj + dynamic_tool_kp_num] = dynamic_tool_kp_future[1] - dynamic_tool_kp_future[0]
            
            # static tool
            static_tool_kp_future = static_tool_kps[n_his+fi:n_his+fi+2]
            static_tool_kp_future = np.stack(static_tool_kp_future, axis=0)
            tool_future[fi, max_nobj + max_ntool : max_nobj + max_ntool + static_tool_kp_num] = static_tool_kp_future[0]
            assert (static_tool_kp_future[1] == static_tool_kp_future[0]).all(), 'static tool should be static'
        
        # load history states
        state_history = np.zeros((n_his, max_nobj + max_ntool * 2, 3), dtype=np.float32)
        for fi in range(n_his):
            # object 
            obj_kp_his = obj_kps[fi]
            obj_kp_his = [obj_kp_his[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
            obj_kp_his = np.concatenate(obj_kp_his, axis=0)
            obj_kp_his = pad(obj_kp_his, max_nobj)
            state_history[fi, :max_nobj] = obj_kp_his
            
            # dynamic tool
            dynamic_tool_kp_his = dynamic_tool_kps[fi]
            dynamic_tool_kp_his = pad(dynamic_tool_kp_his, max_ntool)
            state_history[fi, max_nobj : max_nobj + max_ntool] = dynamic_tool_kp_his
            
            # static tool
            static_tool_kp_his = static_tool_kps[fi]
            static_tool_kp_his = pad(static_tool_kp_his, max_ntool)
            state_history[fi, max_nobj + max_ntool : max_nobj + max_ntool * 2] = static_tool_kp_his
        
        # load masks
        state_mask = np.zeros((max_nobj + max_ntool * 2), dtype=bool)
        state_mask[:obj_kp_num] = True # obj
        state_mask[max_nobj : max_nobj + dynamic_tool_kp_num] = True # dynamic tool
        state_mask[max_nobj + max_ntool : max_nobj + max_ntool + static_tool_kp_num] = True # static tool
        
        obj_mask = np.zeros((max_nobj,), dtype=bool)
        obj_mask[:obj_kp_num] = True
        
        tool_mask = np.zeros((max_nobj + max_ntool * 2,), dtype=bool)
        tool_mask[max_nobj : max_nobj + dynamic_tool_kp_num] = True # dynamic tool
        tool_mask[max_nobj + max_ntool : max_nobj + max_ntool + static_tool_kp_num] = True # static tool
        
        # construct instance information
        p_rigid = np.zeros(max_n, dtype=np.float32)
        p_instance = np.zeros((max_nobj, max_n), dtype=np.float32)
        j_perm = np.random.permutation(instance_num)
        ptcl_cnt = 0
        # sanity check
        assert sum([len(self.fps_idx_list[j]) for j in range(len(self.fps_idx_list))]) == obj_kp_num
        # fill in p_instance
        for j in range(instance_num):
            p_instance[ptcl_cnt:ptcl_cnt + len(self.fps_idx_list[j_perm[j]]), j_perm[j]] = 1
            ptcl_cnt += len(self.fps_idx_list[j_perm[j]])
        
        # construct physics information
        physics_param = self.physics_params[episode_idx]
        physics_param += np.random.uniform(-self.phys_noise, self.phys_noise, size=physics_param.shape)
        physics_param = np.tile(physics_param, (max_nobj, 1)) # (N, phys_dim)
        
        # construct attributes
        attr_dim = 2 # (obj, tool (dynamic_tool + static_tool) )
        attrs = np.zeros((max_nobj + max_ntool * 2, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[max_nobj : max_nobj + dynamic_tool_kp_num, 1] = 1.
        attrs[max_nobj + max_ntool : max_nobj + max_ntool + static_tool_kp_num, 1] = 1.
        
        # state randomness
        state_history += np.random.uniform(-0.01, 0.01, size=state_history.shape)
        
        # TODO: rotation randomness
        random_rot = np.random.uniform(-np.pi, np.pi)
        rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                            [np.sin(random_rot), np.cos(random_rot), 0],
                           [0, 0, 1]], dtype=state_history.dtype) # 2D rotation matrix in xy plane
        state_history = state_history @ rot_mat[None]
        states_delta = states_delta @ rot_mat
        tool_future = tool_future @ rot_mat[None]
        states_delta_future = states_delta_future @ rot_mat[None]
        obj_kp_future = obj_kp_future @ rot_mat[None]
        
        # translation randomness
        random_translation = np.random.uniform(-1, 1, size=3)
        state_history += random_translation[None, None]
        states_delta += random_translation[None]
        tool_future += random_translation[None, None]
        states_delta_future += random_translation[None, None]
        obj_kp_future += random_translation[None, None]
        
        # save graph
        graph = {
            ## N: max_nobj, M: max_ntool
            # input info 
            "state": state_history, # (n_his, N+2M, state_dim)
            "action": states_delta, # (N+2M, state_dim)
            
            # future info
            "tool_future": tool_future, # (n_future-1, N+2M, state_dim)
            "action_future": states_delta_future, # (n_future-1, N+2M, state_dim)
            
            # gt info
            "state_future": obj_kp_future, # (n_future, N, state_dim)
            "state_future_mask": obj_future_mask, # (n_future, )
            
            # attr info
            "attrs": attrs, # (N+2M, attr_dim)
            "p_rigid": p_rigid, # (n_instance, )
            "p_instance": p_instance, # (N, n_instance)
            "physics_param": physics_param, # (N, phys_dim)
            "state_mask": state_mask, # (N+2M, )
            "tool_mask": tool_mask, # (N+2M, )
            "obj_mask": obj_mask, # (N, )
        }
        return graph



