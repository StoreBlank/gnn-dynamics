import numpy as np
import h5py

def load_data(data_names, path):
    hf = h5py.File(path, "r")
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data