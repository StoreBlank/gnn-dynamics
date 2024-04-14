import os
import numpy as np

step_path = '/mnt/sda/data_simple/cloth/episode_79/steps.npy'
steps = np.load(step_path)
print(steps)