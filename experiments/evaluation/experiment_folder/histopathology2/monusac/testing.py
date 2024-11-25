import os
import micro_sam.training as sam_training

# def len_identical():
#     identical = len(os.listdir('/scratch/users/u11644/models/evaluation/monusac_eval/instance_eval/instance_segmentation_with_decoder/inference')) == len(os.listdir('/scratch/users/u11644/data/monusac/monusac_test/complete_labels'))
#     print(identical)
#     print(len(os.listdir('/scratch/users/u11644/models/evaluation/monusac_eval/instance_eval/instance_segmentation_with_decoder/inference')))
#     print(len(os.listdir('/scratch/users/u11644/data/monusac/monusac_test/complete_labels')))
   

# #len_identical()
# def make_experiment_folders(directory,organ_types=None):
#     if organ_types is not None:
#         for organ in organ_types:
#             os.makedirs(os.path.join(directory,f'{organ}', 'amg_eval'))
#             os.makedirs(os.path.join(directory,f'{organ}', 'instance_eval'))
#             os.makedirs(os.path.join(directory,f'{organ}', 'it_prompt_box_eval'))
#             os.makedirs(os.path.join(directory,f'{organ}', 'it_prompt_point_eval'))
#     else:
#         os.makedirs(os.path.join(directory,'complete_dataset', 'amg_eval'))
#         os.makedirs(os.path.join(directory,'complete_dataset', 'instance_eval'))
#         os.makedirs(os.path.join(directory,'complete_dataset', 'it_prompt_box_eval'))
#         os.makedirs(os.path.join(directory,'complete_dataset', 'it_prompt_point_eval'))
# organ_list = ['kidney', 'prostate', 'lung', 'breast']
# make_experiment_folders('/scratch/users/u11644/models/evaluation/monusac_eval/',organ_list)

from monusac import get_monusac_paths
import numpy as np
import os
import imageio
from tqdm import tqdm
def test():
    for split in tqdm(['train', 'test']):
        image_paths, label_paths = get_monusac_paths('/mnt/lustre-grete/usr/u12649/scratch/data/monusac/download/complete_dataset', split)
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
