import os
import numpy as np
import argparse

def get_processed_eef_states(data_dir, epi_idx):
    processed_eef_states = np.load(os.path.join(data_dir, f'episode_{epi_idx}/processed_eef_states.npy'))
    print(f'Episode {epi_idx}: {processed_eef_states}')

def get_steps(data_dir, epi_idx):
    steps = np.load(os.path.join(data_dir, f'episode_{epi_idx}/steps.npy'))
    print(f'Episode {epi_idx}: {steps}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='rope')
    parser.add_argument('--epi_idx', type=int, default=900)
    args = parser.parse_args()
    
    data_dir = f'/mnt/nvme1n1p1/baoyu/data/{args.data_name}'
    
    # get_processed_eef_states(data_dir, epi_idx)
    get_steps(data_dir, args.epi_idx)