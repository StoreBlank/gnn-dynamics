import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append('/home/lzx/workspace/unified_dyn_graph')

import numpy as np
import time
import yaml
import glob
import torch

from env.flex_env import FlexEnv
from utils_env import load_yaml
from gnn.model import DynamicsPredictor
from gnn.utils import set_seed
import cv2
from preprocess.preprocess_rope import extract_eef_points, extract_pushes
from rollout import rollout_dataset
from ipdb import set_trace
# from dataset import construct_edges_from_states, load_dataset
# from utils import rgb_colormap, fps_rad_idx, pad, vis_points, moviepy_merge_video
# from train import truncate_graph


config = load_yaml("../unified_dyn_graph/config/data_gen/gnn_dyn.yaml")
env = FlexEnv(config)
os.makedirs('tem/episode_0', exist_ok=True)
particle_pos_list, eef_states_list, step_list, contact_list = env.reset(dir='tem/episode_0')

u = [0.3, 1.4, -1, 1]
img, particle_pos_list, eef_states_list, step_list, contact_list = env.step(u, dir='tem/episode_0', particle_pos_list=particle_pos_list, eef_states_list=eef_states_list, step_list=step_list, contact_list=contact_list)
u = [-1, 0, 1, 0]
img, particle_pos_list, eef_states_list, step_list, contact_list = env.step(u, dir='tem/episode_0', particle_pos_list=particle_pos_list, eef_states_list=eef_states_list, step_list=step_list, contact_list=contact_list)

np.save(os.path.join('tem/episode_0', 'particles_pos'), particle_pos_list)
np.save(os.path.join('tem/episode_0', 'eef_states.npy'), eef_states_list)
np.save(os.path.join('tem/episode_0', 'steps.npy'), step_list)
np.save(os.path.join('tem/episode_0', 'contact.npy'), contact_list)
cam_intrinsic_params, cam_extrinsic_matrix = env.get_camera_params()
np.save(os.path.join('tem', 'camera_intrinsic_params.npy'), cam_intrinsic_params)
np.save(os.path.join('tem', 'camera_extrinsic_matrix.npy'), cam_extrinsic_matrix)

# set_trace()

env.close()

extract_pushes('tem', 'tem', 0.1, 4, 3)
extract_eef_points('tem')

config = load_yaml('src/config/rope_0119.yaml')
train_config = config['train_config']
dataset_config = config['dataset_config']
dataset_config['data_dir'] = 'tem'
dataset_config['prep_data_dir'] = 'tem'
dataset_config['ratio']['valid'] = [0, 1]
model_config = config['model_config']
epoch = 100
set_seed(train_config['random_seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = 'tem'
checkpoint_dir = os.path.join(train_config['out_dir'], 'checkpoints', 'model_{}.pth'.format(epoch))

model_config['n_his'] = train_config['n_his']
model = DynamicsPredictor(model_config, device).to(device)
model.eval()
model.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))
save_dir_dataset = 'tem'
rollout_dataset(model, device, dataset_config, save_dir_dataset)
