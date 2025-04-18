import os
import numpy as np
import shutil
import random

def merge_data(main_folders, destination_path, material_categories):
    os.makedirs(destination_path, exist_ok=True)
    
    # initialize a list to hold (original_subfolder_path, material) tuples
    subfolders_info = []
    
    # collect all subfolders and their corresponding material
    for i, folder in enumerate(main_folders):
        for subfolder in os.listdir(folder):
            if subfolder.endswith(".npy"):
                continue # skip the camera info
            subfolders_info.append((os.path.join(folder, subfolder), material_categories[i]))
        
    # shuffle the subfolders information list
    np.random.shuffle(subfolders_info)
    
    # rename and merge subfolders
    for new_index, (subfolder_path, material) in enumerate(subfolders_info):
        new_subfolder_name = f"episode_{new_index}"
        new_subfolder_path = os.path.join(destination_path, new_subfolder_name)
        # os.makedirs(new_subfolder_path, exist_ok=False)
        
        # copy the entire subfolder to the new destination with a new name
        print(f"Copying {subfolder_path} to {new_subfolder_path}...")
        shutil.copytree(subfolder_path, new_subfolder_path)
        
        # create a file to store the material category
        with open(os.path.join(new_subfolder_path, "material.txt"), "w") as f:
            f.write(material)
        
        print(f"Done with {new_subfolder_path} with material {material}.")

def copy_random_subfolders(source_dir, target_dir, num_folders):
    # Make sure the source directory exists
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory {source_dir} does not exist")
    
    # List all subdirectories in the source directory
    subfolders = [os.path.join(source_dir, name) for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, name))]
    print(f"Found {len(subfolders)} subfolders in {source_dir}")
    
    # Check if the number of subfolders is sufficient
    if len(subfolders) < num_folders:
        raise ValueError("The number of subfolders is less than the number of folders you want to copy.")
    
    # Randomly select 'num_folders' subfolders
    selected_folders = random.sample(subfolders, num_folders)
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy the selected folders to the target directory and rename them
    for i, folder in enumerate(selected_folders):
        new_folder_name = f"episode_{i}"
        new_folder_path = os.path.join(target_dir, new_folder_name)
        
        print(f"Copying {folder} to {new_folder_path}...")
        shutil.copytree(folder, new_folder_path)

def add_material_txt(dir, material):
    epi_start = 0
    epi_end = 1500
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(dir, f"episode_{i}")
        with open(os.path.join(epi_dir, "material.txt"), "w") as f:
            f.write(material)
        print(f"Done with {epi_dir} with material {material}.")

if __name__ == "__main__":
    main_folders = [
        "/mnt/sda/data_simple/carrots",
        "/mnt/sda/data_simple/cloth",
        "/mnt/sda/data_simple/rope"
    ]
    material_categories = ["granular", "cloth", "rope"]
    # destination_path = "/mnt/sda/data_simple/mixed"
    # merge_data(main_folders, destination_path, material_categories)
    
    # subset_dir = "/mnt/sda/data_simple/mixed_subset"
    # copy_random_subfolders(destination_path, subset_dir, 10)
    
    dir = "/mnt/sda/data_simple/carrots_mixed_1"
    material = material_categories[0]
    add_material_txt(dir, material)
    
    
    