import torch
import os
from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from pannuke import get_pannuke_dataset
from pannuke import get_pannuke_loader
import h5py
from torch_em.transform.label import PerObjectDistanceTransform
import tifffile
import micro_sam.training as sam_training
import numpy as np
from tqdm import tqdm
import time


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

    train_loader = get_pannuke_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=2,
        folds=["fold_1"],
        num_workers=16,
        download=False,
        shuffle=True,
        label_transform=label_transform,
        raw_transform=raw_transform,
        label_dtype=torch.float32,
        sampler=sampler,
        ndim=2,
    )
    val_loader = get_pannuke_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        folds=["fold_2"],
        num_workers=16,
        download=False,
        shuffle=True,
        label_transform=label_transform,
        raw_transform=raw_transform,
        label_dtype=torch.float32,
        sampler=sampler,
        ndim=2,
    )
    test_loader = get_pannuke_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        folds=["fold_3"],
        num_workers=16,
        download=False,
        shuffle=True,
        raw_transform=raw_transform,
        label_dtype=torch.float32,
        sampler=sampler,
        ndim=2,
    )

    return train_loader, val_loader, test_loader


def visualize_images(data_path):
    train_loader, val_loader, test_loader = get_dataloaders(patch_shape=(1, 512, 512), data_path=data_path)

    # let's visualize train loader first
    check_loader(train_loader, 8, plt=True, save_path="./fig.png")



# def load_pannuke_dataset(path):
#     counter = 1
#     _path = os.path.join(path, 'loaded_dataset', 'complete_dataset')
#     _, __, he_loader = get_dataloaders(patch_shape=(1,256,256), data_path=path)
#     print(len(he_loader))
#     image_output_path = os.path.join(_path, 'images')
#     label_output_path = os.path.join(_path, 'labels')
#     os.makedirs(image_output_path, exist_ok=True)
#     os.makedirs(label_output_path, exist_ok=True)
#     for image, label in he_loader:
#         image_array = image.numpy()
#         label_array = label.numpy()
#         squeezed_image = image_array.squeeze()
#         label_data = label_array.squeeze()
#         transposed_image_array = squeezed_image.transpose(1,2,0)
#         #print(f'image {counter:04} shape: {np.shape(transposed_image_array)}, label {counter:04} shape: {np.shape(label_data)}')
#         assert np.shape(transposed_image_array)[0] == 256, f'Shape error in image {counter:04}'
#         assert np.shape(transposed_image_array)[1] == 256, f'Shape error in image {counter:04}'
#         assert np.shape(transposed_image_array)[2] == 3, f'Shape error in image {counter:04}'
#         assert np.shape(label_data)[0] == 256, f'Shape error in label {counter:04}'
#         assert np.shape(label_data)[0] == 256, f'Shape error in label {counter:04}'
#         tif_image_output_path = os.path.join(image_output_path,f'{counter:04}.tiff')
#         tifffile.imwrite(tif_image_output_path, transposed_image_array)
#         tif_label_output_path = os.path.join(label_output_path,f'{counter:04}.tiff')
#         tifffile.imwrite(tif_label_output_path, label_data)
#         counter+=1
#     print('All images have a confirmed shape of (256, 256, 3) and all labels have a shape of (256,256)')
# from pannuke import get_pannuke_paths
# #load_pannuke_dataset('/mnt/lustre-grete/usr/u12649/scratch/data/pannuke')
# #get_pannuke_dataset('/mnt/lustre-grete/usr/u12649/scratch/data/pannuke_test', (256, 256), folds=['fold_1'], download=True)
# data_paths = get_pannuke_paths('/mnt/lustre-grete/usr/u12649/scratch/data/pannuke_test',['fold_1'],True)

# def examine_pannuke():
#     data_paths = get_pannuke_paths('/mnt/lustre-grete/usr/u12649/scratch/data/pannuke_test',['fold_1'],True)
#     h5_path = data_paths[0]
#     with h5py.File(h5_path, 'r') as f:
#         images = f['images']
#         np_images = np.array(images)
#         tp_images = np_images.transpose(1,0,2,3)
#         print(tp_images.shape)
#         for image in tqdm(tp_images):
#             print(f'Image datatype: {image.dtype}, max value: {np.max(image)}, min value: {np.min(image)}')

# examine_pannuke()
