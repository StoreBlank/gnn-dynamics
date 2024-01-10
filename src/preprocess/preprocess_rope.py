import os
import glob
import json
import numpy as np
import open3d as o3d
import torch

from dgl.geometry import farthest_point_sampler
from utils import quaternion_to_rotation_matrix, fps_rad_idx, pad

def extract_kp(data_dir, episode_idx, start_frame, end_frame):
    # obtain object keypoints
    obj_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    obj_ptcl_start, obj_ptcl_end = obj_ptcl[start_frame], obj_ptcl[end_frame]
    obj_kp = np.stack([obj_ptcl_start, obj_ptcl_end], axis=0)
    
    # obtain tool keypoints
    tool_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/eef_states.npy"))
    tool_kps = np.zeros((tool_ptcl.shape[0], 3))
    for i in range(tool_ptcl.shape[0]):
        tool_pos = tool_ptcl[i, 0:3]
        tool_quat = tool_ptcl[i, 6:10]
        tool_rot = quaternion_to_rotation_matrix(tool_quat)
        tool_kps[i] = tool_pos + np.dot(tool_rot, np.array([0, 0, 1.]))
    tool_kp = np.stack([tool_kps[start_frame], tool_kps[end_frame]], axis=0)
    
    # TODO: generalize to multiple tools
    
    return obj_kp, tool_kp

def extract_kp_single_frame(data_dir, episode_idx, frame_idx):
    # obtain object keypoints
    obj_ptcls = np.load(os.path.join(data_dir, f"episode_{episode_idx}/particles_pos.npy"))
    # print(f"epi_idx {episode_idx}, frame_idx {frame_idx}, obj_ptcls: {obj_ptcls.shape}")
    obj_ptcl = obj_ptcls[frame_idx]
    obj_kp = np.array([obj_ptcl])
    
    # obtain tool keypoints
    tool_ptcl = np.load(os.path.join(data_dir, f"episode_{episode_idx}/eef_states.npy"))
    tool_kps = np.zeros((tool_ptcl.shape[0], 3))
    for i in range(tool_ptcl.shape[0]):
        tool_pos = tool_ptcl[i, 0:3]
        tool_quat = tool_ptcl[i, 6:10]
        tool_rot = quaternion_to_rotation_matrix(tool_quat)
        tool_kps[i] = tool_pos + np.dot(tool_rot, np.array([0, 0, 1.]))
    tool_kp = np.array([tool_kps[frame_idx]])
    
    return obj_kp, tool_kp

"""
Preprocess data to save the following:
    - frame_pairs: a directory containing the start-end frame pairs for each push.
        - {epi_idx}_{push_idx}.txt: save the push pairs for each frame: (n_his - 1, curr, n_future)
    - phys_range.txt: a file containing the min and max of the physical parameters.
    - metadata.txt: a file containing the metadata of the preprocessed data.
"""

def extract_pushes(data_dir, save_dir, dist_thresh, n_his, n_future):
    """
    Args:
        data_dir (str): directory of the raw data
        save_dir (str): directory to save the processed data
        dist_thresh (float): distance threshold to determine a push pair
        n_his (int): number of frames to look back
        n_future (int): number of frames to look forward
    """
    
    frame_idx_dir = os.path.join(save_dir, "frame_pairs")
    os.makedirs(frame_idx_dir, exist_ok=True)
    
    # calculate the number of episodes folder in the data directory
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Preprocessing starts. Number of episodes: {num_episodes}")
    
    phys_params = []
    
    for epi_idx in range(num_episodes):
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{epi_idx}/camera_0/*_color.jpg"))))
        print(f"Processing episode {epi_idx}, num_frames: {num_frames}")
        
        # load info
        steps = np.load(os.path.join(data_dir, f"episode_{epi_idx}/steps.npy"))
        
        eef_states = np.load(os.path.join(data_dir, f"episode_{epi_idx}/eef_states.npy"))
        assert eef_states.shape[0] == num_frames
        eef_pos = np.zeros((num_frames, 3))
        for i in range(num_frames):
            eef_pos_0 = eef_states[i, 0:3]
            eef_quat = eef_states[i, 6:10]
            eef_rot = quaternion_to_rotation_matrix(eef_quat)
            eef_pos[i] = eef_pos_0 + np.dot(eef_rot, np.array([0, 0, 1.]))
        
        physics_path = os.path.join(data_dir, f"episode_{epi_idx}/property_params.json")
        with open(physics_path, "r") as f:
            properties = json.load(f)
        phys_param = np.array([
            # properties['particle_radius'],
            # properties['num_particles'],
            properties['length'],
            properties['thickness'],
            properties['dynamic_friction'],
            properties['cluster_spacing']
        ]).astype(np.float32)
        phys_params.append(phys_param)
        
        # get start-end pairs
        frame_idxs = []
        cnts = [0]
        cnt = 0
        for fj in range(num_frames):
            curr_step = None
            for si in range(len(steps) - 1):
                if fj >= steps[si] and fj <= steps[si + 1] - 2:
                    curr_step = si
                    break
            else:
                continue # this frame is not valid
            assert curr_step is not None
        
            curr_frame = fj
            start_frame = steps[curr_step]
            end_frame = steps[curr_step + 1] - 2
            
            # search backward (n_his)
            eef_particles_curr = eef_pos[curr_frame]
            frame_traj = [curr_frame]
            fi = fj
            while fi >= start_frame:
                eef_particles_fi = eef_pos[fi]
                x_curr, z_curr = eef_particles_curr[0], eef_particles_curr[2]
                x_fi, z_fi = eef_particles_fi[0], eef_particles_fi[2]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)
                if dist_curr >= dist_thresh:
                    frame_traj.append(fi)
                    eef_particles_curr = eef_particles_fi
                fi -= 1
                if len(frame_traj) == n_his:
                    break
            else: 
                # pad to n_his
                frame_traj = frame_traj + [frame_traj[-1]] * (n_his - len(frame_traj))
            frame_traj = frame_traj[::-1]
            
            # search forward (n_future)
            eef_particles_curr = eef_pos[curr_frame]
            fi = fj
            while fi <= end_frame:
                eef_particles_fi = eef_pos[fi]
                x_curr, z_curr = eef_particles_curr[0], eef_particles_curr[2]
                x_fi, z_fi = eef_particles_fi[0], eef_particles_fi[2]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)
                if dist_curr >= dist_thresh or (fi == end_frame and dist_curr >= 0.75 * dist_thresh):
                    frame_traj.append(fi)
                    eef_particles_curr = eef_particles_fi
                fi += 1
                if len(frame_traj) == n_his + n_future:
                    cnt += 1
                    break
            else:
                # When assuming quasi-static, we can pad to n_his + n_future
                frame_traj = frame_traj + [frame_traj[-1]] * (n_his + n_future - len(frame_traj))
                cnt += 1
            
            frame_idxs.append(frame_traj)

            # push_centered
            if fj == end_frame:
                cnts.append(cnt)
                frame_idxs = np.array(frame_idxs)
                np.savetxt(os.path.join(frame_idx_dir, f"{epi_idx}_{curr_step}.txt"), frame_idxs, fmt="%d")
                print(f"episode {epi_idx}, push {curr_step} has {cnts[curr_step+1]-cnts[curr_step]} pushes.")
                frame_idxs = []
    
    # save phys_params stat
    phys_params = np.stack(phys_params, axis=0)
    phys_params_max = np.max(phys_params, axis=0)
    phys_params_min = np.min(phys_params, axis=0)
    phys_params_range = np.stack([phys_params_min, phys_params_max], axis=0)
    print(f"phys_params_range: {phys_params_range}")
    np.savetxt(os.path.join(save_dir, "phys_range.txt"), phys_params_range)
    
def extract_tool_points(data_dir, tool_names, tool_scale, tool_sample_points=100, max_ntool=100, vis=False):
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Preprocessing tool starts. Number of episodes: {num_episodes}")
    
    n_tools = len(tool_names)
    print(f"There are {n_tools} tools: {tool_names}")
    
    for epi_idx in range(num_episodes):
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{epi_idx}/camera_0/*_color.jpg"))))
        print(f"Processing episode {epi_idx}, num_frames: {num_frames}")
        
        # load the tool mesh and get the initial point cloud
        tool_mesh_dir = os.path.join(data_dir, f"geometries/tools")
        tool_surface_points_list = []
        for i in range(n_tools):
            # convert mesh to point cloud
            tool_mesh_path = os.path.join(tool_mesh_dir, f'{tool_names[i]}.obj')
            tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
            
            tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 10000)
            # tool_surface = o3d.geometry.TriangleMesh.sample_points_uniformly(tool_mesh, sample_points)
            if vis:
                o3d.visualization.draw_geometries([tool_surface])
            
            # rescale the point cloud
            tool_surface.points = o3d.utility.Vector3dVector(np.asarray(tool_surface.points) * tool_scale[i])
            tool_surface_points = np.asarray(tool_surface.points)
            
            # fps sampling
            particle_tensor = torch.from_numpy(tool_surface_points).float().unsqueeze(0)
            fps_idx_tensor = farthest_point_sampler(particle_tensor, tool_sample_points)[0]
            fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)
            
            # downsample to uniform radius
            downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
            fps_radius = 0.1
            _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
            fps_idx_2 = fps_idx_2.astype(np.int32)
            fps_idx = fps_idx_1[fps_idx_2]
            print(f'tool {tool_names[i]} has {fps_idx.shape[0]} sample points.')
            
            # obtain fps tool surface points
            tool_surface_points = tool_surface_points[fps_idx]
            # pad to max_ntool
            tool_surface_points = pad(tool_surface_points, max_ntool)
            if vis:
                o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tool_surface_points))])
            
            tool_surface_points_list.append(tool_surface_points)
        
        # obtain the tool points for each frame
        all_tool_points = np.zeros((n_tools, num_frames, max_ntool, 3))
        for i in range(n_tools):
            tool_i_frame_points = []
            for frame_idx in range(num_frames):
                # load init tool surface
                tool_surface_f = o3d.geometry.PointCloud()
                tool_surface_f.points = o3d.utility.Vector3dVector(tool_surface_points_list[i])
                
                # load the pos and orientation of the tool
                tool_points_path = os.path.join(data_dir, f"episode_{epi_idx}/{tool_names[i]}_states.npy")
                tool_points = np.load(tool_points_path)
                
                tool_ori = tool_points[frame_idx, 3:]
                tool_rot = quaternion_to_rotation_matrix(tool_ori)
                tool_surface_f.rotate(tool_rot)
                
                tool_pos = tool_points[frame_idx, :3]
                tool_surface_f.translate(tool_pos)
                
                tool_i_frame_points.append(np.asarray(tool_surface_f.points))
            
            all_tool_points[i] = np.stack(tool_i_frame_points, axis=0)
        
        # save the tool points to the data_dir
        for idx, tool_name in enumerate(tool_names):
            np.save(os.path.join(data_dir, f"episode_{epi_idx}/{tool_name}_points.npy"), all_tool_points[idx])
            print(f"Tool {tool_name} points saved to {data_dir}/episode_{epi_idx}/{tool_name}_points.npy")
        

if __name__ == "__main__":
    # i = 4
    data_name = "rope"
    data_dir_list = [
        f"/mnt/sda/data/{data_name}"
    ]
    save_dir_list = [
        f"/mnt/sda/preprocess/{data_name}"
    ]
    dist_thresh = 0.5 #5cm
    n_his = 4
    n_future = 3
    
    for data_dir, save_dir in zip(data_dir_list, save_dir_list):
        if os.path.isdir(data_dir):
            os.makedirs(save_dir, exist_ok=True)
            print("================extract_pushes================")
            extract_pushes(data_dir, save_dir, dist_thresh, n_his, n_future)
            print("==============================================")
        # save metadata
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
            f.write(f'{dist_thresh},{n_future},{n_his}')