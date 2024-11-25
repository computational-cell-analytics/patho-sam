from pannuke import get_pannuke_paths
import numpy as np
import os
import imageio
from tqdm import tqdm


# def test():
#     folds = ["fold_1", "fold_2", "fold_3"]
#     for fold in tqdm(folds):
#         image_paths = get_pannuke_paths('/mnt/lustre-grete/usr/u12649/scratch/data/pannuke', folds)
#         max_value = []
#         for image in tqdm(image_paths):
            
    #         image = imageio.imread(image)
    #         # image = image.astype(np.float32)
    #         print(f'Image datatype: {image.dtype}')
    #         unique_values = np.unique(image)
    #         print("Max value:", max(unique_values))
    #         print("Min value:", min(unique_values))
    #         max_value.append(max(unique_values))
    # print(f'{max(max_value)} was the maximum value of the given images')
    

# test()
from pannuke import get_pannuke_data

def explore(path):
    npy = np.load(path)
    max_value = []
    for image in tqdm(npy):
        unique_values = np.unique(image)
        print("Max value:", max(unique_values))
        print("Min value:", min(unique_values))
        print(f'Image datatype: {image.dtype}')
        max_value.append(max(unique_values))
    print(f'Max value: {max(max_value)}')


explore('/mnt/lustre-grete/usr/u12649/scratch/data/pannuke/fold_1/Fold 1/images/fold1/images.npy')