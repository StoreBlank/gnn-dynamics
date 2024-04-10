import os
import numpy as np
import shutil

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

if __name__ == "__main__":
    main_folders = [
        "/mnt/sda/data_simple/carrots",
        "/mnt/sda/data_simple/cloth",
        "/mnt/sda/data_simple/rope"
    ]
    material_categories = ["granular", "cloth", "rope"]
    destination_path = "/mnt/sda/data_simple/mixed"
    merge_data(main_folders, destination_path, material_categories)
    
    
    
    