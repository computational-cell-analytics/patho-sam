import torch
import os
from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.histopathology.monusac import get_monusac_loader
from torch_em.transform.label import PerObjectDistanceTransform
import numpy as np
import micro_sam.training as sam_training
import tifffile
from PIL import Image



def get_dataloaders(patch_shape, data_path):
    """This returns the pannuke data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/histopathology/pannuke.py
    It will automatically download the pannuke data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25
    )
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)

    train_loader = get_monusac_loader(
        path=data_path,
        patch_shape=patch_shape,
        split = "train",
        batch_size=1,
        organ_type=['kidney'],
        download=True,
        offsets=None,
        boundaries=False,
        binary=False,
    )
    val_loader = get_monusac_loader(
        path=data_path,
        patch_shape=patch_shape,
        split = "test",
        batch_size=1,
        organ_type=['kidney'],
        download=True,
        offsets=None,
        boundaries=False,
        binary=False,
    )
    return train_loader, val_loader


def load_and_save_monusac(label_output_path, image_output_path):
    train_loader, val_loader = get_dataloaders(patch_shape=(1,512,512),data_path='/scratch/users/u11644/data/monusac/download')
    counter = 1      
    for image,label in train_loader:
       image_array = image.numpy()
       label_array = label.numpy()
       #print(f'Image {counter:04} original shape: {np.shape(image_array)}')
       squeezed_image = image_array.squeeze()
       squeezed_label = label_array.squeeze()
       transposed_image_array = squeezed_image.transpose(1,2,0)
       print(f'Image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
       tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
       tifffile.imwrite(tif_image_output_path, transposed_image_array)
       tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
       tifffile.imwrite(tif_label_output_path, squeezed_label)
       counter+=1
    for image,label in val_loader:
       image_array = image.numpy()
       label_array = label.numpy()
       print(f'Image {counter:04} original shape: {np.shape(image_array)}')
       squeezed_image = image_array.squeeze()
       squeezed_label = label_array.squeeze()
       transposed_image_array = squeezed_image.transpose(1,2,0)
       print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
       tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
       tifffile.imwrite(tif_image_output_path, transposed_image_array)
       tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
       tifffile.imwrite(tif_label_output_path, squeezed_label)
       counter+=1


load_and_save_monusac('/scratch/users/u11644/data/monusac/monusac_test/by_organs/kidney/labels','/scratch/users/u11644/data/monusac/monusac_test/by_organs/kidney/images')

# 14
# 51
# 52
# 65
# 81
# 82