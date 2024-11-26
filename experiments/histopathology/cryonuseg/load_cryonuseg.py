from cryonuseg import get_cryonuseg_loader
import os
import tifffile
import numpy as np
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training


def get_dataloaders(patch_shape, data_path, rater):
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)
    loader = get_cryonuseg_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        rater=rater,
        download=True,
        raw_transform=raw_transform,
        sampler=sampler,
    )
    return loader


def load_cryonuseg_dataset(path):
    counter = 1
    _path = os.path.join(path, 'test', 'loaded_dataset', 'complete_dataset')
    for rater in ['b1', 'b2', 'b3']:
        rater_loader = get_dataloaders(patch_shape=(1, 512, 512), data_path=path, rater=rater)
        image_output_path = os.path.join(_path, 'images')
        label_output_path = os.path.join(_path, 'labels')
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)
        for image, label in rater_loader:
            image_array = image.numpy()
            label_array = label.numpy()
            squeezed_image = image_array.squeeze()
            label_data = label_array.squeeze()            
            transposed_image_array = squeezed_image.transpose(1, 2, 0)
            print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(label_data)}')
            tif_image_output_path = os.path.join(image_output_path, f'{counter:04}.tiff')
            tifffile.imwrite(tif_image_output_path, transposed_image_array)
            tif_label_output_path = os.path.join(label_output_path, f'{counter:04}.tiff')
            tifffile.imwrite(tif_label_output_path, label_data)
            counter += 1


load_cryonuseg_dataset('/mnt/lustre-grete/usr/u12649/scratch/data/cryonuseg')