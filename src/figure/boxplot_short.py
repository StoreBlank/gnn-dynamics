import os 
import glob
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection

font = {'family' : 'Times New Roman'}
matplotlib.rc('font', **font)


def process_short_data(data_dir, epi_start, epi_end):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    all_short_error, all_short_error_baseline = [], []
    for i in range(epi_start, epi_end):
        # short erros
        short_dir = os.path.join(data_dir, f"{i}", "short")
        # read the file
        short_error = np.loadtxt(os.path.join(short_dir, "error_4.txt"))
        short_error_baseline = np.loadtxt(os.path.join(short_dir, "error_baseline_4.txt"))
        all_short_error.append(short_error)
        all_short_error_baseline.append(short_error_baseline)
    return all_short_error, all_short_error_baseline
        
def boxplot(data, save_dir, max_step=30, name="rope"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    all_data = []
    for i in range(len(data)):
        short_error = data[i][max_step] / 10 # dm to m
        all_data.append(short_error)
    
    # # plot the boxplot
    # fig, ax = plt.subplots(figsize=(3.8, 2.5))
    # bplot = ax.boxplot(all_data, patch_artist=True, vert=True)
    # ax.set_xticklabels(["Unified \n GNN", "Seperate \n GNN", "Ours w/o \n Adaptation", "Ours"], fontsize=10)
    # ax.set_ylabel("CD (m)", fontsize=10)
    # # ax.set_xlabel("Model")
    # # ax.set_title("Rope short Error", fontsize=12)
    # ax.xaxis.set_label_coords(0.5, -0.1)
    # ax.yaxis.set_label_coords(-0.2, 0.5)
    
    # # fill with colors
    # colors = ["lightyellow", "pink", "lightblue", "lightgreen"]
    # for patch, color in zip(bplot["boxes"], colors):
    #     patch.set_facecolor(color)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f"{name}_short_error.png"), dpi=300)
    # plt.savefig(os.path.join(save_dir, f"{name}_short_error.pdf"), dpi=300)
    
    # plot the boxplot
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    bplot = ax.boxplot(all_data, patch_artist=True, vert=True)
    ax.set_xticklabels(["Unified \n GNN", "Ours"], fontsize=10)
    ax.set_ylabel("CD (m)", fontsize=10)
    # ax.set_xlabel("Model")
    # ax.set_title("Rope short Error", fontsize=12)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.2, 0.5)
    
    # fill with colors
    colors = ["lightyellow", "lightgreen"]
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_short_error.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{name}_short_error.pdf"), dpi=300)
    
    
    
    
if __name__ == "__main__":
    
    # # rope
    # epi_start = 900
    # epi_end = 1000
    # max_step = 30
    # y_axis = -0.25
    # data_dirs = [
    #     "/mnt/sda/adaptigraph/vis/rollout-all_0414-model_100/mixed_rope",
    #     # "/mnt/sda/adaptigraph/vis_main/rollout-rope_0119_no_physics-model_100/rope",
    #     # "/mnt/sda/adaptigraph/vis_main/rollout-rope_0119-model_100-constant/rope",
    #     "/mnt/sda/adaptigraph/vis_main/rollout-rope_0119-model_100/rope"
    # ]
    
    # name = "rope"
    # save_dir = f"/home/baoyu/2024/gnn-dynamics/src/figure/{name}/short"
    # all_short_error = []
    # for i in range(len(data_dirs)):
    #     short_error, short_error_baseline = process_short_data(data_dirs[i], epi_start, epi_end)
    #     all_short_error.append(short_error)
    # boxplot(all_short_error, save_dir, max_step, name)
    
    # granular
    epi_start = 1350
    epi_end = 1500
    max_step = 30
    y_axis = -0.35
    data_dirs = [
        "/mnt/sda/adaptigraph/vis/rollout-all_0414-model_100/mixed_granular",
        # "/mnt/sda/adaptigraph/vis_main/rollout-granular_0127_no_physics-model_100/granular",
        # "/mnt/sda/adaptigraph/vis_main/rollout-granular_0127-model_100-constant/granular",
        "/mnt/sda/adaptigraph/vis_main/rollout-granular_0127-model_100/granular"
    ]
    
    name = "granular"
    save_dir = f"/home/baoyu/2024/gnn-dynamics/src/figure/{name}/short"
    all_short_error = []
    for i in range(len(data_dirs)):
        short_error, short_error_baseline = process_short_data(data_dirs[i], epi_start, epi_end)
        all_short_error.append(short_error)
    boxplot(all_short_error, save_dir, max_step, name)
    
    # cloth
    epi_start = 900
    epi_end = 1000
    max_step = 11
    y_axis = -0.2
    data_dirs = [
        "/mnt/sda/adaptigraph/vis/rollout-all_0414-model_100/mixed_cloth",
        # "/mnt/sda/adaptigraph/vis_main/rollout-cloth_0127_no_physics-model_100/cloth",
        # "/mnt/sda/adaptigraph/vis_main/rollout-cloth_0127-model_100/cloth",
        "/mnt/sda/adaptigraph/vis_main/rollout-cloth_0127-model_100_ours/cloth"
    ]
    
    name = "cloth"
    save_dir = f"/home/baoyu/2024/gnn-dynamics/src/figure/{name}/short"
    all_short_error = []
    for i in range(len(data_dirs)):
        short_error, short_error_baseline = process_short_data(data_dirs[i], epi_start, epi_end)
        all_short_error.append(short_error)
    boxplot(all_short_error, save_dir, max_step, name)