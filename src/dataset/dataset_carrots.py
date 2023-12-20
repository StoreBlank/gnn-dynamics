import os
import glob
import json
import numpy as np

import torch 
from torch.utils.data import Dataset

from dataset.utils import extract_kp_single_frame, fps_rad_idx, pad
from dgl.geometry import farthest_point_sampler

def construct_edges_from_states(states, adj_thresh, mask, eef_mask, no_self_edge=False):  # helper function for construct_graph
    # :param states: (B, N, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N) torch tensor, true when index is a valid particle
    # :param eef_mask: (B, N) torch tensor, true when index is a valid eef particle
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor
    B, N, state_dim = states.shape
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    dis = torch.sum((s_sender - s_receiv)**2, -1)
    mask_1 = mask[:, :, None].repeat(1, 1, N)
    mask_2 = mask[:, None, :].repeat(1, N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    eef_mask_1 = eef_mask[:, :, None].repeat(1, 1, N)
    eef_mask_2 = eef_mask[:, None, :].repeat(1, N, 1)
    eef_mask_12 = eef_mask_1 * eef_mask_2
    dis[eef_mask_12] = 1e10  # avoid eef to eef relations
    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # remove self edge
    if no_self_edge:
        self_edge_mask = torch.eye(N, device=states.device, dtype=states.dtype)[None, :, :]
        adj_matrix = adj_matrix * (1 - self_edge_mask)

    # add topk constraints
    topk = 5
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix
    
    n_rels = adj_matrix.sum(dim=(1,2))
    # print(n_rels.shape, (mask * 1.0).sum(-1).mean().item(), n_rels.mean().item())
    n_rel = n_rels.max().long().item()
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    return Rr, Rs

class CarrotsDynDataset(Dataset):
    def __init__(
        self,
        data_dir, 
        prep_save_dir, 
        phase, 
        ratios, 
        dist_thresh,
        fps_radius_range, # radius for fps sampling
        adj_thres_range, # threshold for constructing edges (not used here)
        n_future,
        n_his,
        max_n, # max number of objects
        max_nobj, # max number of object points
        max_neef, # max number of eef points
        max_nR, # max number of relations (not used here)
        phys_noise = 0.0,
        canonical = False,
    ):
        self.phase = phase
        self.data_dir = data_dir
        print(f'Setting up CarrotsDynDataset')
        print(f'Setting up {phase} dataset, data_dir: {len(data_dir)}')
        
        self.n_his = n_his
        self.n_future = n_future
        self.dist_thresh = dist_thresh
        
        self.max_n = max_n
        self.max_nobj = max_nobj
        self.max_neef = max_neef
        self.fps_radius_range = fps_radius_range
        self.phys_noise = phys_noise
        
        self.obj_kypts_paths = []
        self.eef_kypts_paths = []
        self.physics_paths = []
        
        # load kypts paths
        num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
        print(f"Found num_episodes: {num_episodes}")
        frame_count = 0
        for episode_idx in range(num_episodes):
            n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_particles.npy"))))
            obj_kypts_paths = [os.path.join(data_dir, f"episode_{episode_idx}/camera_0", f"{frame_idx}_particles.npy") for frame_idx in range(n_frames)]
            eef_kypts_paths = [os.path.join(data_dir, f"episode_{episode_idx}/camera_0", f"{frame_idx}_endeffector.npy") for frame_idx in range(n_frames)]
            physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property.json")
            self.obj_kypts_paths.append(obj_kypts_paths)
            self.eef_kypts_paths.append(eef_kypts_paths)
            self.physics_paths.append(physics_path)
            frame_count += n_frames
        print(f'Found {frame_count} frames in {data_dir}')
        
        # load data pairs
        save_dir = os.path.join(prep_save_dir, data_dir.split('/')[-1])
        pairs_path = os.path.join(save_dir, 'frame_pairs')
        self.pair_lists = []
        for episode_idx in range(num_episodes):
            prev_pair_len = len(self.pair_lists)
            n_pushes = len(list(glob.glob(os.path.join(pairs_path, f'{episode_idx}_*.txt'))))
            for push_idx in range(n_pushes):
                frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}_{push_idx}.txt'))
                if len(frame_pairs.shape) == 1: continue
                episodes = np.ones((frame_pairs.shape[0], 1)) * episode_idx
                pairs = np.concatenate([episodes, frame_pairs], axis=1)
                self.pair_lists.append(pairs)
            self.pair_lists = np.array(self.pair_lists)
            print('pair lists shape: ', self.pair_lists.shape)
            print(f'Found {len(self.pair_lists)} frame pairs in {pairs_path}')
        
        # load phys_params 
        physics_range = np.loadtxt(os.path.join(save_dir, 'phys_range.txt')).astype(np.float32)
        self.physics_params = []
        for episode_idx in range(num_episodes):
            physics_path = self.physics_paths[episode_idx]
            assert os.path.join(self.data_dir, f"episode_{episode_idx}/property.json") == physics_path
            with open(physics_path, "r") as f:
                properties = json.load(f)
            phys_param = np.array([
                # properties['particle_radius'],
                # properties['num_particles'],
                properties['rand_scale'],
                properties['blob_r'],
                properties['num_granule'],
                properties['dynamic_friction'],
                properties['mass']
            ]).astype(np.float32)
            physics_param = (physics_param - physics_range[0]) / (physics_range[1] - physics_range[0] + 1e-6)  # normalize
            self.physics_params.append(physics_param)
        self.physics_params = np.stack(self.physics_params, axis=0) # (N, phys_dim)
        
        # take ratio to split pari lists
        self.ratio = ratios
        print(f'Taking ratio {self.ratio} of {len(self.pair_lists)} eef keypoint files')
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
    
    def __getitem__(self, idx):
        max_n = self.max_n
        max_nobj = self.max_nobj
        max_neef = self.max_neef
        n_his = self.n_his
        n_future = self.n_future
        fps_radius_range = self.fps_radius_range
        
        episode_idx = self.pair_lists[i, 0].astype(np.int32)
        pair = self.pair_lists[i, 1:].astype(np.int32)
        
        # get history keypoints
        obj_kps, eef_kps = [], []
        for i in range(len(pair)):
            frame_idx = pair[i]
            obj_kp, eef_kp = extract_kp_single_frame(self.data_dir, episode_idx, frame_idx)
            obj_kps.append(obj_kp)
            eef_kps.append(eef_kp)
        
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
        eef_kp = np.stack(eef_kps[n_his-1:n_his+1], axis=0) # (2, 1, 3)
        eef_kp_num = eef_kp.shape[1]
        start_delta = np.zeros((max_nobj + max_neef, obj_kp_start.shape[-1]), dtype=np.float32)
        start_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]
        
        # load future states
        obj_kp_future = np.zeros((n_future, max_nobj, obj_kp_start.shape[-1]), dtype=np.float32)
        obj_future_mask = np.ones(n_future).astype(bool) # (n_future, )
        for fi in range(n_future):
            obj_kp_fu = obj_kps[n_his + fi]
            obj_kp_fu = [obj_kp_fu[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
            obj_kp_fu = np.concatenate(obj_kp_fu, axis=0) # (N, 3)
            obj_kp_fu = pad(obj_kp_fu, max_nobj)
            obj_kp_future[fi] = obj_kp_fu
        
        # load future eef keypoints
        eef_future = np.zeros((n_future - 1, max_nobj + max_neef, obj_kp_start.shape[-1]), dtype=np.float32)
        states_delta_future = np.zeros((n_future - 1, max_nobj + max_neef, obj_kp_start.shape[-1]), dtype=np.float32)
        for fi in range(n_future - 1):
            eef_kp_future = eef_kps[n_his+fi:n_his+fi+2]
            eef_kp_future = np.stack(eef_kp_future, axis=0) # (2, 1, 3)
            eef_kp_future = pad(eef_kp_future, max_neef, dim=1) # (2, max_neef, 3)
            eef_future[fi, max_nobj : max_nobj + eef_kp_num] = eef_kp_future[0]
            states_delta_future[fi, max_nobj : max_nobj + eef_kp_num] = eef_kp_future[1] - eef_kp_future[0]
        
        # load history states
        state_history = np.zeros((n_his, max_nobj + max_neef, obj_kp_start.shape[-1]), dtype=np.float32)
        for fi in range(n_his):
            obj_kp_his = obj_kps[fi]
            obj_kp_his = [obj_kp_his[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
            obj_kp_his = np.concatenate(obj_kp_his, axis=0)
            obj_kp_his = pad(obj_kp_his, max_nobj)
            state_history[fi, :max_nobj] = obj_kp_his

            eef_kp_his = eef_kps[fi]
            eef_kp_his = pad(eef_kp_his, max_neef)
            state_history[fi, max_nobj:] = eef_kp_his
        
        # load masks
        state_mask = np.zeros((max_nobj + max_neef), dtype=bool)
        state_mask[max_nobj : max_nobj + eef_kp_num] = True
        state_mask[:obj_kp_num] = True
        
        eef_mask = np.zeros((max_nobj + max_neef), dtype=bool)
        eef_mask[max_nobj : max_nobj + eef_kp_num] = True
        
        obj_mask = np.zeros((max_nobj,), dtype=bool)
        obj_mask[:obj_kp_num] = True
        
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
        # TODO: what are these attributes?
        attr_dim = 2
        attrs = np.zeros((max_nobj + max_neef, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[max_nobj : max_nobj + eef_kp_num, 1] = 1.
        
        ## add randomness TODO: ?
        # state randomness
        state_history += np.random.uniform(-0.01, 0.01, size=state_history.shape)
        
        # rotation randomness
        random_rot = np.random.uniform(-np.pi, np.pi)
        rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                            [np.sin(random_rot), np.cos(random_rot), 0],
                           [0, 0, 1]], dtype=state_history.dtype) # 2D rotation matrix in xy plane
        state_history = state_history @ rot_mat[None]
        states_delta = states_delta @ rot_mat
        eef_future = eef_future @ rot_mat[None]
        states_delta_future = states_delta_future @ rot_mat[None]
        obj_kp_future = obj_kp_future @ rot_mat[None]
        
        # translation randomness
        random_translation = np.random.uniform(-1, 1, size=3)
        state_history += random_translation[None, None]
        states_delta += random_translation[None]
        eef_future += random_translation[None, None]
        states_delta_future += random_translation[None, None]
        obj_kp_future += random_translation[None, None]
        
        # save graph
        graph = {
            ## N: max_nobj, M: max_neef
            # input info 
            "state": state_history, # (n_his, N+M, state_dim)
            "action": states_delta, # (N+M, state_dim)
            
            # future info
            "eef_future": eef_future, # (n_future-1, N+M, state_dim)
            "action_future": states_delta_future, # (n_future-1, N+M, state_dim)
            
            # gt info
            "state_future": obj_kp_future, # (n_future, N, state_dim)
            "state_future_mask": obj_future_mask, # (n_future, )
            
            # attr info
            "attr": attrs, # (N+M, attr_dim)
            "p_rigid": p_rigid, # (n_instance, )
            "p_instance": p_instance, # (N, n_instance)
            "physics_param": physics_param, # (N, phys_dim)
            "state_mask": state_mask, # (N+M, )
            "eef_mask": eef_mask, # (N+M, )
            "obj_mask": obj_mask, # (N, )
        }
        return graph
        
        
        
        
        
        
        
        
        
        