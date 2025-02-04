import os
from glob import glob
from natsort import natsorted


import torch
import torch.utils.data as data_util
import torch_em
from torch_em.data import MinInstanceSampler, datasets
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb


def histopathology_identity(x, ensure_rgb=True):
    """Identity transform.
    Inspired from 'micro_sam/training/util.py' -> 'identity' function.

    This ensures to skip data normalization when finetuning SAM.
    Data normalization is performed within the model to SA-1B data statistics
    and should thus be skipped as a preprocessing step in training.
    """
    if ensure_rgb:
        x = to_rgb(x)

    return x


def _get_train_val_split(ds, val_fraction=0.2):
    """Dataset split. This splits the provided dataset used for training into a train and val set.
    """
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds


def get_dataloaders(patch_shape, batch_size, data_path=None, dataset=None, images_dir=None, masks_dir=None):

    if dataset is not None:
        assert masks_dir is None and images_dir is None, "Provide either a dataset name or directories for custom data"
    else:
        assert dataset is None, "Provide either a dataset name or directories for custom data"
    label_dtype = torch.float32

    # Expected raw and label transforms for training
    raw_transform = histopathology_identity
    label_transform = PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=True,
        min_size=10,
    )

    if dataset == 'nuclick':
        ds = datasets.get_nuclick_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            split="Train",
            sampler=MinInstanceSampler(min_num_instances=2, min_size=10),
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    if dataset is None:
        raw_paths = natsorted(glob(os.path.join(images_dir, "*")))
        label_paths = natsorted(glob(os.path.join(masks_dir, "*")))

        assert len(raw_paths) == len(label_paths)

        ds = torch_em.default_segmentation_dataset(
            raw_paths=raw_paths,
            raw_key=None,
            label_paths=label_paths,
            label_key=None,
            patch_shape=patch_shape,
            ndim=2,
            with_channels=True,
            sampler=MinInstanceSampler(min_num_instances=2),
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
            )

    else:
        raise NotImplementedError

    train_ds, val_ds = _get_train_val_split(ds=ds)

    train_loader = torch_em.get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=1, shuffle=True, num_workers=16)

    return train_loader, val_loader
