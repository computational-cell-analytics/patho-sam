from lizard2 import get_lizard_paths
import numpy as np
import os
import imageio
from tqdm import tqdm
import h5py

def test():
    volume_paths = get_lizard_paths('/mnt/lustre-grete/usr/u12649/scratch/data/tnbc')
    max_value = []
    for volume in tqdm(volume_paths):
        with h5py.File(volume, 'r') as f:
            img = f['raw']
            image = np.asarray(img)
            print(f'Image datatype: {image.dtype}')
            unique_values = np.unique(image)
            # print("Max value:", max(unique_values))
            # print("Min value:", min(unique_values))
            max_value.append(max(unique_values))
    print(f'{max(max_value)} was the maximum value of the given images')
    

test()


