import os
from glob import glob
from natsort import natsorted

import torch_em
from torch_em.data.datasets import util


def custom_transform(x, y):
    return x, y


def histopathology_identity(x):
    """Identity transform.
    Inspired from 'micro_sam/training/util.py' -> 'identity' function.

    This ensures to skip data normalization when finetuning SAM.
    Data normalization is performed within the model to SA-1B data statistics
    and should thus be skipped as a preprocessing step in training.
    """

    return x


def get_loader(path, patch_shape, batch_size, **kwargs):
    print(path)
    image_paths = natsorted(glob(os.path.join(path, "images", "*.tiff")))
    label_paths = natsorted(glob(os.path.join(path, "semantic", "*.tiff")))
    print("length of image_paths, label paths: ", len(image_paths), len(label_paths))
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)



