import os 
import shutil
import random

def mixed_two_folders(folder_path_1, folder_path_2, new_folder_path):
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)
    
    # move subfolders from folder_path_1 and folder_path_2 to new_folder_path
    for folder in [folder_path_1, folder_path_2]:
        for subfolder in os.listdir(folder):
            shutil.move(os.path.join(folder, subfolder), new_folder_path)
            print(f"Moved {folder, subfolder} to {new_folder_path}")
    
    
def change_file_names(folder_path):
    for folder_name in os.listdir(folder_path):
        if folder_name.startswith("episode_"):
            episode_num = int(folder_name.split("_")[1])
            new_episode_num = episode_num + 1000 #TODO
            new_folder_name = f"episode_{new_episode_num}"
            
            old_folder_path = os.path.join(folder_path, folder_name)
            new_folder_path = os.path.join(folder_path, new_folder_name)
            
            os.rename(old_folder_path, new_folder_path)
            print(f"Renamed {folder_name} to {new_folder_name}")

def move_rename_folders(source_folder, new_folder_path):
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)
    
    # Get a list of all subfolders in the source folder
    subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    
    # Split the list into folders with and without 'camera_1'
    folders_with_camera = [f for f in subfolders if os.path.isdir(os.path.join(source_folder, f, 'camera_0'))]
    folders_without_camera = [f for f in subfolders if not os.path.isdir(os.path.join(source_folder, f, 'camera_0'))]

    # Combine the lists, placing folders with 'camera_1' at the end
    ordered_folders = folders_without_camera + folders_with_camera

    # Rename and move each subfolder
    for i, subfolder in enumerate(ordered_folders):
        old_path = os.path.join(source_folder, subfolder)
        new_path = os.path.join(new_folder_path, f"episode_{i}")
        shutil.move(old_path, new_path)
    
if __name__ == "__main__":
    # folder_path_1 = "/mnt/nvme1n1p1/baoyu/data_simple/carrots"
    # folder_path_2 = "/mnt/nvme1n1p1/baoyu/data_simple/carrots_poly"
    # change_file_names(folder_path_2)
    # mixed_two_folders(folder_path_1, folder_path_2, "/mnt/nvme1n1p1/baoyu/data_simple/carrots_mixed_1")
    
    foler_path_1 = "/mnt/nvme1n1p1/baoyu/data_simple/carrots_mixed_2"
    new_folder_path = "/mnt/nvme1n1p1/baoyu/data_simple/carrots_mixed_1"
    move_rename_folders(foler_path_1, new_folder_path)
    
    
    