from puma import get_puma_loader
from glob import glob
import os
from natsort import natsorted
import tifffile
import numpy as np
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training
from scipy.io import loadmat

def get_dataloaders(patch_shape, data_path):
    
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)
    loader = get_puma_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        annotations='nuclei',
        download=True,
        raw_transform=raw_transform,
        sampler=sampler,
        #offsets=None,
        #boundaries=False,
        #binary=False,
    )
    return loader


def load_puma_dataset(path):
    counter = 1
    _path = os.path.join(path, 'loaded_dataset', 'complete_dataset')
    puma_loader = get_dataloaders(patch_shape=(1,512,512), data_path=path)

    image_output_path = os.path.join(_path, 'images')
    label_output_path = os.path.join(_path, 'labels')
    
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    for image, label in puma_loader:
        image_array = image.numpy()
        label_array = label.numpy()
        squeezed_image = image_array.squeeze()
        label_data = label_array.squeeze()
        
            
        # label_data = new_label.numpy()
        # num_zeros = (label_data == 0).sum().item()
        # print(f"Number of 0s in the new label {counter:04}: {num_zeros}")
        # #breakpoint()
        
        transposed_image_array = squeezed_image #removed transposition
        print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(label_data)}')
        tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
        tifffile.imwrite(tif_image_output_path, transposed_image_array)
        tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
        tifffile.imwrite(tif_label_output_path, label_data)
        counter+=1

load_puma_dataset('/mnt/lustre-grete/usr/u12649/scratch/data/puma')