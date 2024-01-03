import argparse
import numpy as np
from datetime import datetime

### build arguments
parser = argparse.ArgumentParser()

"""
GNN config
"""
parser.add_argument('--nf_relation', type=int, default=150)
parser.add_argument('--nf_particle', type=int, default=150)
parser.add_argument('--nf_effect', type=int, default=150)
parser.add_argument('--state_normalize', type=int, default=1)

'''
system and run config
'''
parser.add_argument('--data_dir', type=str, default='/mnt/sda/data/granular_sweeping_dustpan')
parser.add_argument('--prep_save_dir', type=str, default='/mnt/sda/preprocess')
parser.add_argument('--out_dir', type=str, default='/mnt/sda/logs/granular_sweeping_dustpan')
parser.add_argument('--adj_thresh_min', type=float, default=0.09)
parser.add_argument('--adj_thresh_max', type=float, default=0.11)

# evaluate
parser.add_argument('--data_name', type=str, default='granular_sweeping_dustpan')
parser.add_argument('--checkpoint_name', type=str, default='granular_sweeping_dustpan')
parser.add_argument('--checkpoint_epoch', type=int, default=10) #100
parser.add_argument('--adj_thresh', type=float, default=0.1)

parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--verbose', type=int, default=0)


def gen_args():
    args = parser.parse_args()

    args.mean_p = np.array([0.50932539, 0.11348496, 0.49837578])
    args.std_p = np.array([0.06474939, 0.04888084, 0.05906044])

    args.mean_d = np.array([-0.00284736, 0.00286124, -0.00130389])
    args.std_d = np.array([0.01755744, 0.01663332, 0.01677678])

    return args
