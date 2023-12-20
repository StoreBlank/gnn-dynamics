import os
import time
import sys
import numpy as np

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.base_config import gen_args
from gnn.model import DynamicsPredictor
from gnn.utils import set_seed, umeyama_algorithm
from dataset.dataset_carrots import CarrotsDataset, construct_edges_from_states


def train_carrots(out_dir, data_dirs, prep_save_dir=None, material='carrots', ratios=None):
    torch.autograd.set_detect_anomaly(True)
    args = gen_args()