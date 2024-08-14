import os
import glob
import json
import numpy as np
import open3d as o3d
import torch


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

    for epi_idx in range(num_episodes):
        # load particle
        particles_pos = np.load(os.path.join(data_dir, f"episode_{epi_idx}/particles_pos.npy"))
        num_frames = particles_pos.shape[0]
        # num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{epi_idx}/camera_0/*_color.jpg"))))
        print(f"Processing episode {epi_idx}, num_frames: {num_frames}")

        # load info
        steps = np.load(os.path.join(data_dir, f"episode_{epi_idx}/steps.npy"))

        eef_states = np.load(os.path.join(data_dir, f"episode_{epi_idx}/eef_states.npy"))
        assert eef_states.shape[0] == num_frames

        eef_pos = eef_states[:, :2]

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
                x_curr, y_curr = eef_particles_curr[0], eef_particles_curr[1]
                x_fi, y_fi = eef_particles_fi[0], eef_particles_fi[1]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (y_curr - y_fi) ** 2)
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
                x_curr, y_curr = eef_particles_curr[0], eef_particles_curr[1]
                x_fi, y_fi = eef_particles_fi[0], eef_particles_fi[1]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (y_curr - y_fi) ** 2)
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

def extract_eef_points(data_dir):
    # calculate the number of episodes folder in the data directory
    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
    print(f"Preprocessing starts. Number of episodes: {num_episodes}")

    # granular eef info
    # x_max: +-15
    eef_point_pos = np.array([
        [-15, 0],
        [0, 0],
        [15, 0],
    ])
    n_eef_points = eef_point_pos.shape[0]

    for epi_idx in range(num_episodes):
        # load particle
        particles_pos = np.load(os.path.join(data_dir, f"episode_{epi_idx}/particles_pos.npy"))
        num_frames = particles_pos.shape[0]
        # num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{epi_idx}/camera_0/*_color.jpg"))))
        print(f"Processing episode {epi_idx}, num_frames: {num_frames}")

        # load the eef states
        eef_states = np.load(os.path.join(data_dir, f"episode_{epi_idx}/eef_states.npy"))
        assert eef_states.shape[0] == num_frames

        # extract eef points
        processed_eef_states = np.zeros((num_frames, n_eef_points, 2))
        for frame_idx in range(num_frames):
            eef_state = eef_states[frame_idx]
            eef_pos_0 = eef_state[0:2]
            eef_angle = eef_state[2]
            eef_rot = np.array([
                [np.cos(eef_angle), -np.sin(eef_angle)],
                [np.sin(eef_angle), np.cos(eef_angle)],
            ])
            for j in range(n_eef_points):
                eef_pos = eef_pos_0 + np.dot(eef_rot, eef_point_pos[j])
                processed_eef_states[frame_idx, j] = eef_pos

        # save the processed eef states
        np.save(os.path.join(data_dir, f"episode_{epi_idx}/processed_eef_states.npy"), processed_eef_states)

if __name__ == "__main__":
    data_dir_list = [
        "../data/merging_L"
    ]
    save_dir_list = [
        "../data/preprocess/merging_L"
    ]
    dist_thresh = 10. # 1cm
    n_his = 4
    n_future = 3

    for data_dir, save_dir in zip(data_dir_list, save_dir_list):
        if os.path.isdir(data_dir):
            os.makedirs(save_dir, exist_ok=True)
            print("================extract_pushes================")
            extract_pushes(data_dir, save_dir, dist_thresh, n_his, n_future)
            print("==============================================")
            print("================extract eef================")
            extract_eef_points(data_dir)
            print("==============================================")
        # save metadata
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
            f.write(f'{dist_thresh},{n_future},{n_his}')
