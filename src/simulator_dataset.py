import os
import glob
import json
import numpy as np

import torch 
from torch.utils.data import Dataset

from dgl.geometry import farthest_point_sampler
from utils import pad, pad_torch, fps_rad_idx

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

def load_dataset(dataset, material_config, phase='train'):
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

    physics_params = []
    for episode_idx in range(num_episodes):
        physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property_params.json")
        with open(physics_path) as f:
            properties = json.load(f)
        
        physics_params_episode = {}

        for material_name in dataset["materials"]:
            material_params = material_config[material_name]['physics_params']

            used_params = []
            for item in material_params:
                if item['name'] in properties.keys() and item['use']:
                    range_min = item['min']
                    range_max = item['max']
                    used_params.append((properties[item['name']] - range_min) / (range_max - range_min + 1e-6))
            
            used_params = np.array(used_params).astype(np.float32)
            physics_params_episode[material_name] = used_params
        physics_params.append(physics_params_episode)

    return pair_lists, physics_params  

class SimDataset(Dataset):
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

        self.pair_lists = []
        self.physics_params = []

        self.materials = {}

        for i, dataset in enumerate(dataset_config['datasets']):
            print(f'Setting up dataset {dataset["name"]} at {dataset["data_dir"]}')
            materials_list = dataset['materials']
            
            pair_lists, physics_params = load_dataset(dataset, material_config, phase)
            pair_lists = np.concatenate([np.ones((pair_lists.shape[0], 1)) * i, pair_lists], axis=1)
            for k in physics_params[0].keys():
                self.materials[k] = physics_params[0][k].shape[0]
            print('Length of dataset is', len(pair_lists))

            self.pair_lists.extend(pair_lists)
            self.physics_params.append(physics_params)  # [dataset_idx][episode_idx][material_name][param_idx]

        self.pair_lists = np.array(self.pair_lists) 
        
        num_episodes = len(list(glob.glob(os.path.join(dataset_config['datasets'][0]["data_dir"], f"episode_*"))))
        data_dir = dataset_config['datasets'][0]["data_dir"]
        
        # save all particles and tool states
        self.all_particle_pos = []
        self.all_tool_states = []
        for episode_idx in range(num_episodes):
            particles_pos = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
            tool_states = np.load(os.path.join(data_dir, f"episode_{episode_idx}/processed_eef_states.npy"))
            # print(f'episode {episode_idx}: tool_states shape: {tool_states.shape}.')
            self.all_particle_pos.append(particles_pos) 
            self.all_tool_states.append(tool_states)
        
    def __len__(self):
        return len(self.pair_lists)
    
    def __getitem__(self, i):

        dataset_idx = self.pair_lists[i][0].astype(int)
        episode_idx = self.pair_lists[i][1].astype(int)
        pair = self.pair_lists[i][2:].astype(int)
        
        assert dataset_idx == 0, 'only support single dataset'
        
        n_his = self.dataset_config['n_his']
        n_future = self.dataset_config['n_future']

        dataset_config = self.dataset_config['datasets'][dataset_idx]
        # max_n = dataset_config['max_n']
        # max_tool = dataset_config['max_tool']
        # max_nobj = dataset_config['max_nobj']
        # max_ntool = dataset_config['max_ntool']
        # max_nR = dataset_config['max_nR']
        # fps_radius_range = dataset_config['fps_radius_range']
        # adj_radius_range = dataset_config['adj_radius_range']
        # state_noise = dataset_config['state_noise'][self.phase]
        phys_noise = dataset_config['phys_noise'][self.phase]
        
        # get history keypoints
        obj_kps, tool_kps = [], []
        for i in range(len(pair)):
            frame_idx = pair[i]
            # obj_kp: (1, num_obj_points, 3)
            # tool_kp: (num_tool_points, 3)
            # obj_ptcls = self.all_particle_pos[episode_idx]
            # obj_kp, tool_kp = extract_kp_single_frame(dataset_config['data_dir'], episode_idx, frame_idx)
            # print(obj_kp.shape, tool_kp.shape) 
            
            obj_kp = self.all_particle_pos[episode_idx][frame_idx] # (num_obj_points, 3)
            tool_kp = self.all_tool_states[episode_idx][frame_idx] # (num_tool_points, 3)
            
            # print(obj_kp.shape, tool_kp.shape)
            
            obj_kps.append(obj_kp)
            tool_kps.append(tool_kp) # (7, num_tool_points, 3)
        
        obj_kp_start = obj_kps[n_his-1]
        # obj_kp_end = obj_kps[-1]
        obj_kp_num = obj_kp_start.shape[0]

        tool_kp_start = tool_kps[n_his-1][0, [0, 2]]
        tool_kp_end = tool_kps[-1][0, [0, 2]]
        tool_kp_num = tool_kp.shape[1]

        state = np.zeros((2000, 3))
        state_mask = np.zeros(2000)
        state[:obj_kp_num, :] = obj_kp_start
        state_mask[:obj_kp_num] = 1
        action = np.concatenate([tool_kp_start, tool_kp_end], axis=0)

        # construct physics information
        physics_param = self.physics_params[dataset_idx][episode_idx]  # dict
        for material_name in dataset_config['materials']:
            if material_name not in physics_param.keys():
                raise ValueError(f'Physics parameter {material_name} not found in {dataset_config["data_dir"]}')
            physics_param[material_name] += np.random.uniform(-phys_noise, phys_noise, 
                    size=physics_param[material_name].shape)

        # save graph
        graph = {
            ## N: max_nobj, M: max_ntool
            # input info 
            "state": state, # (N, state_dim)
            "action": action, # (N, action_dim)
            "state_mask": state_mask, # (N,)
        }

        for material_name, material_dim in self.materials.items():
            if material_name in physics_param.keys():
                graph[material_name + '_physics_param'] = physics_param[material_name]
            else:
                graph[material_name + '_physics_param'] = torch.zeros(material_dim, dtype=torch.float32)

        return graph



