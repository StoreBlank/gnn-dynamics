import os
import glob
import json
import numpy as np

"""
Preprocess carrots data to save the following:
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
        actions = np.load(os.path.join(data_dir, f"episode_{epi_idx}/actions.npy"))
        steps = np.load(os.path.join(data_dir, f"episode_{epi_idx}/steps.npy"))
        eef_pos = np.load(os.path.join(data_dir, f"episode_{epi_idx}/eef_pos.npy"))
        particles_pos = np.load(os.path.join(data_dir, f"episode_{epi_idx}/particles_pos.npy"))
        
        physics_path = os.path.join(data_dir, f"episode_{epi_idx}/property.json")
        with open(physics_path, "r") as f:
            properties = json.load(f)
        phys_param = np.array([
            properties['particle_radius'],
            properties['num_particles'],
            properties['granular_scale'],
            properties['num_granular'],
            properties['distribution_r'],
            properties['dynamic_friction'],
            properties['granular_mass']
        ]).astype(np.float32)
        phys_params.append(phys_param)
        
        # get start-end pairs
        frame_idxs = []
        cnt = 0
        for fj in range(2, num_frames):
            curr_step = None
            for si in range(len(steps) - 1):
                """
                steps[si]: start frame of the push
                steps[si + 1] - 2: end frame of the push
                steps[si + 1] - 1: the final render for the push (not consider as a push)
                steps[si + 1]: the start frame of the next push
                """
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
                x_curr, z_curr = eef_particles_curr[0], eef_particles_curr[1]
                x_fi, z_fi = eef_particles_fi[0], eef_particles_fi[1]
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
                x_curr, z_curr = eef_particles_curr[0], eef_particles_curr[1]
                x_fi, z_fi = eef_particles_fi[0], eef_particles_fi[1]
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
                frame_idxs = np.array(frame_idxs)
                np.savetxt(os.path.join(frame_idx_dir, f"{epi_idx}_{curr_step}.txt"), frame_idxs, fmt="%d")
                print(f"episode {epi_idx}, push {curr_step} has {cnt} pushes.")
                frame_idxs = []
    
    # save phys_params stat
    phys_params = np.stack(phys_params, axis=0)
    phys_params_max = np.max(phys_params, axis=0)
    phys_params_min = np.min(phys_params, axis=0)
    phys_params_range = np.stack([phys_params_min, phys_params_max], axis=0)
    print(f"phys_params_range: {phys_params_range}")
    np.savetxt(os.path.join(save_dir, "phys_range.txt"), phys_params_range)

if __name__ == "__main__":
    data_dir_list = [
        "/mnt/sda/data/carrots_5"
    ]
    save_dir_list = [
        "/mnt/sda/preprocess/carrots_5"
    ]
    dist_thresh = 0.05
    n_his = 4
    n_future = 3
    for data_dir, save_dir in zip(data_dir_list, save_dir_list):
        if os.path.isdir(data_dir):
            os.makedirs(save_dir, exist_ok=True)
            extract_pushes(data_dir, save_dir, dist_thresh, n_his, n_future)
        # save metadata
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
            f.write(f'{dist_thresh},{n_future},{n_his}')