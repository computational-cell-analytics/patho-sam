import os
from torch_em.data import MinInstanceSampler
from monuseg import get_monuseg_loader
import numpy as np
import micro_sam.training as sam_training
import tifffile


def get_dataloaders(patch_shape, data_path, organ_type):
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)

    train_loader = get_monuseg_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        split="train",
        organ_type=organ_type,
        download=True,
        offsets=None,
        boundaries=False,
        binary=False,
        raw_transform=raw_transform,
        sampler=sampler
    )
    if organ_type is None:
        val_loader = get_monuseg_loader(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            split="test",
            organ_type=organ_type,
            download=True,
            offsets=None,
            boundaries=False,
            binary=False,
            raw_transform=raw_transform,
            sampler=sampler
         )
    else:
        val_loader = None      
    return train_loader, val_loader

def load_and_save_monuseg(directory, organ_type=None):
    data_path = '/mnt/lustre-grete/usr/u12649/scratch/data/monuseg/download/complete_dataset'
    os.makedirs(data_path, exist_ok=True)

    if organ_type is not None:
        organ_combination = ''
        for organ in organ_type:
            organ_combination = organ_combination + '_' + organ
        image_output_path = os.path.join(directory, organ_combination, 'images') 
        label_output_path = os.path.join(directory, organ_combination, 'labels') 
    else:
        image_output_path = os.path.join(directory, 'test ', 'complete_dataset', 'images') 
        label_output_path = os.path.join(directory,  'test', 'complete_dataset', 'labels')
    train_loader, val_loader = get_dataloaders(patch_shape=(1, 512, 512), data_path=data_path, organ_type=organ_type)
    counter = 1
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    for image,label in train_loader:
        image_array = image.numpy()
        label_array = label.numpy()
        #print(f'Image {counter:04} original shape: {np.shape(image_array)}')
        squeezed_image = image_array.squeeze()
        squeezed_label = label_array.squeeze()
        transposed_image_array = squeezed_image.transpose(1, 2, 0)
        print(f'Image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
        tif_image_output_path = os.path.join(image_output_path, f'{counter:04}.tiff')
        tifffile.imwrite(tif_image_output_path, transposed_image_array)
        tif_label_output_path = os.path.join(label_output_path, f'{counter:04}.tiff')
        tifffile.imwrite(tif_label_output_path, squeezed_label)
        counter += 1
    if organ_type is None:
        for image, label in val_loader:
            image_array = image.numpy()
            label_array = label.numpy()
            #print(f'Image {counter:04} original shape: {np.shape(image_array)}')
            squeezed_image = image_array.squeeze()
            squeezed_label = label_array.squeeze()
            transposed_image_array = squeezed_image.transpose(1, 2, 0)
            print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
            tif_image_output_path = os.path.join(image_output_path, f'{counter:04}.tiff')
            tifffile.imwrite(tif_image_output_path, transposed_image_array)
            tif_label_output_path = os.path.join(label_output_path, f'{counter:04}.tiff')
            tifffile.imwrite(tif_label_output_path, squeezed_label)
            counter += 1


load_and_save_monuseg('/mnt/lustre-grete/usr/u12649/scratch/data/monuseg/loaded_data', organ_type=None)
