import os
import tifffile
import numpy as np
from create_dataloaders import get_dataloaders


def load_pannuke_dataset(path):
    counter = 1
    _path = os.path.join(path, 'loaded_dataset', 'complete_dataset')
    _, __, test_loader = get_dataloaders(patch_shape=(1,256,256), data_path=path)
    image_output_path = os.path.join(_path, 'images')
    label_output_path = os.path.join(_path, 'labels')
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    for image, label in test_loader:
        image_array = image.numpy()
        label_array = label.numpy()
        squeezed_image = image_array.squeeze()
        label_data = label_array.squeeze()
        transposed_image_array = squeezed_image.transpose(1,2,0)
        print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(label_data)}')
        tif_image_output_path = os.path.join(image_output_path, f'{counter:04}.tiff')
        tifffile.imwrite(tif_image_output_path, transposed_image_array)
        tif_label_output_path = os.path.join(label_output_path, f'{counter:04}.tiff')
        tifffile.imwrite(tif_label_output_path, label_data)
        counter += 1


load_pannuke_dataset('/mnt/lustre-grete/usr/u12649/scratch/data/pannuke')
