import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append('/home/lzx/workspace/unified_dyn_graph')

import numpy as np
import torch
import pygame

from env.pymunk_T import T_Sim
from utils_env import load_yaml
from gnn.model import DynamicsPredictor
from gnn.utils import set_seed
from preprocess.preprocess_T import extract_pushes, extract_eef_points
from rollout_T import rollout_dataset


config = load_yaml("../unified_dyn_graph/config/data_gen/pushing_T.yaml")
env = T_Sim(config['env'])
os.makedirs('tem/episode_0', exist_ok=True)
particle_pos_list, eef_states_list, step_list, contact_list = env.reset(dir='tem/episode_0')

pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

img = env.render()
img = np.transpose(img, (1, 0, 2))
img_str = img.tobytes()
surface = pygame.image.fromstring(img_str, img.shape[:2], 'RGB')
screen.blit(surface, (0, 0))
pygame.display.flip()
