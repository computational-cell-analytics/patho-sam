import os
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.histopathology.monusac import get_monusac_loader
import numpy as np
import micro_sam.training as sam_training
import tifffile


def get_dataloaders(patch_shape, data_path, organ_type):
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler(min_num_instances=3)

    train_loader = get_monusac_loader(
        path=data_path,
        patch_shape=patch_shape,
        split="train",
        batch_size=1,
        organ_type=organ_type,
        download=True,
        offsets=None,
        boundaries=False,
        binary=False,
        raw_transform=raw_transform,
        sampler=sampler
    )
    val_loader = get_monusac_loader(
        path=data_path,
        patch_shape=patch_shape,
        split="test",
        batch_size=1,
        organ_type=organ_type,
        download=True,
        offsets=None,
        boundaries=False,
        binary=False,
        raw_transform=raw_transform,
        sampler=sampler
    )
    return train_loader, val_loader


def load_and_save_monusac(directory, organ_type=None):
    data_path = '/mnt/lustre-grete/usr/u12649/scratch/data/monusac/download/complete_dataset'
    os.makedirs(data_path, exist_ok=True)
    if organ_type is not None:
        organ_combination = ''
        for organ in organ_type:
            organ_combination = organ_combination + '_' + organ
        image_output_path = os.path.join(directory, organ_combination, 'images') 
        label_output_path = os.path.join(directory, organ_combination, 'labels') 
    else:
        image_output_path = os.path.join(directory, 'complete_dataset', 'images') 
        label_output_path = os.path.join(directory, 'complete_dataset', 'labels')
    train_loader, val_loader = get_dataloaders(patch_shape=(1, 512, 512), data_path=data_path, organ_type=organ_type)       
    counter = 1
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    assert os.listdir(image_output_path) == [], 'Images are loaded already'
    assert os.listdir(label_output_path) == [], 'Labels are loaded already'
    for loader in [val_loader, train_loader]:
        for image, label in loader:
            image_array = image.numpy()
            label_array = label.numpy()
            squeezed_image = image_array.squeeze()
            squeezed_label = label_array.squeeze()
            transposed_image_array = squeezed_image.transpose(1, 2, 0)
            print(f'Image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(squeezed_label)}')
            tif_image_output_path = os.path.join(image_output_path, f'{counter:04}.tiff')
            tifffile.imwrite(tif_image_output_path, transposed_image_array)
            tif_label_output_path = os.path.join(label_output_path, f'{counter:04}.tiff')
            tifffile.imwrite(tif_label_output_path, squeezed_label)
            counter += 1


load_and_save_monusac('/mnt/lustre-grete/usr/u12649/scratch/data/monusac/')