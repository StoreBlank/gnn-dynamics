import os
import sys
import glob
import tqdm
import numpy as np
import open3d as o3d

tool_type = 'gripper_sym_rod_robot_v4_surf_nocorr_full_normal_keyframe=16'
tool_name_list = ["gripper_l", "gripper_r"]
tool_geom_path = "geometries/tools"
tool_dim = [92, 92]



def main(vis=True):
    tool_list = []
    for i in range(len(tool_name_list)):
        tool_mesh = o3d.io.read_triangle_mesh(os.path.join(tool_geom_path, f"{tool_name_list[i]}.stl"))
        tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 10000)
        tool_list.append((tool_mesh, tool_surface))
        if vis:
            # visualize tool surface
            o3d.visualization.draw_geometries([tool_surface])
        
    write_frames = False
    write_gt_state = False
    visualize = False
    
    # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    # print(cd)
    
    rollout_dir = '/mnt/sda/robocook/perception'
    os.makedirs(rollout_dir, exist_ok=True)
    
    data_root = '/mnt/sda/robocook/raw/raw_data/gripper_sym_rod_robot_v4'
    dir_list = sorted(glob.glob(os.path.join(data_root, "*")))
    episode_len = 5
    
    start_idx = 0
    n_vids = len(dir_list)
    for i in range(start_idx, int(start_idx+n_vids)):
        vid_idx = str(i).zfill(3)
        print(f'========== Video {vid_idx} ==========')
        
        bag_path = os.path.join(
            data_root,
            f"ep_{str(i // episode_len).zfill(3)}",
            f"seq_{str(i % episode_len).zfill(3)}",
        )
        # print(f"bag_path: {bag_path}")
        
        # bag_list = sorted(
        #     glob.glob(os.path.join(bag_path, "*.bag")),
        #     key=lambda x: float(os.path.basename(x)[:-4]),
        # )
        
        # rollout_path = os.path.join(rollout_dir, f"{vid_idx}")
        # image_path = os.path.join(rollout_path, "images")
        # os.system("mkdir -p " + rollout_path)
        # if write_frames:
        #     os.system("mkdir -p " + f"{rollout_path}/frames")
        # if write_gt_state:
        #     os.system("mkdir -p " + f"{image_path}")
        
        # state_seq = []
        # pcd_dense_prev, pcd_sparse_prev = None, None
        # is_moving_back = False
        # last_dist = float("inf")
        # start_frame = 0
        # step_size = 1
        # # for j in tqdm(
        # #     range(start_frame, len(bag_list), step_size), desc=f"Video {vid_idx}"
        # # ):
        




if __name__ == "__main__":
    main()