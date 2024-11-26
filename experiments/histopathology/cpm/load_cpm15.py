from cpm import get_cpm_loader
import os
import tifffile
import numpy as np
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training


def get_dataloaders(patch_shape, data_path, data_choice):
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)
    loader = get_cpm_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        download=False,
        raw_transform=raw_transform,
        sampler=sampler,
        data_choice=data_choice
    )
    return loader


def load_cpm_dataset(path):
    for data in ['cpm17', 'cpm15']:
        counter = 1
        _path = os.path.join(path, 'test', data, 'loaded_dataset', 'complete_dataset')
        os.makedirs(_path, exist_ok=True)
        data_path = os.path.join(path, data)
        cpm_loader = get_dataloaders(patch_shape=(1, 512, 512), data_path=data_path, data_choice=data)
        image_output_path = os.path.join(_path, 'images')
        label_output_path = os.path.join(_path, 'labels')
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)
        for image, label in cpm_loader:
            image_array = image.numpy()
            label_array = label.numpy()
            squeezed_image = image_array.squeeze()
            label_data = label_array.squeeze()
            transposed_image = squeezed_image.transpose(1, 2, 0)
            print(f'image {counter:04} shape: {np.shape(transposed_image)}, label {counter:04} shape: {np.shape(label_data)}')
            tif_image_output_path = os.path.join(image_output_path, f'{counter:04}.tiff')
            tifffile.imwrite(tif_image_output_path, transposed_image)
            tif_label_output_path = os.path.join(label_output_path, f'{counter:04}.tiff')
            tifffile.imwrite(tif_label_output_path, label_data)
            counter += 1


load_cpm_dataset('/mnt/lustre-grete/usr/u12649/scratch/data')
