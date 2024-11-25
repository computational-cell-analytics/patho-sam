from lynsec import get_lynsec_paths
import numpy as np
import os
import imageio
from tqdm import tqdm


def test():
    image_paths, label_paths = get_lynsec_paths('/mnt/lustre-grete/usr/u12649/scratch/data/lynsec', 'ihc')
    max_value = []
    for image in tqdm(image_paths):
        image = imageio.imread(image)
        # image = image.astype(np.float32)
        print(f'Image datatype: {image.dtype}')
        unique_values = np.unique(image)
        # print("Max value:", max(unique_values))
        # print("Min value:", min(unique_values))
        max_value.append(max(unique_values))
    print(f'{max(max_value)} was the maximum value of the given images')
    

test()


