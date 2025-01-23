import os

import torch
import torch.utils.data as data_util

import torch_em
from torch_em.data import MinInstanceSampler, datasets
from torch_em.transform.label import PerObjectDistanceTransform

from patho_sam.training import histopathology_identity


def _get_train_val_split(ds, val_fraction=0.2):
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds


def get_specialist_dataset(path, patch_shape, split_choice, dataset):
    # Important stuff for dataloaders.
    label_dtype = torch.float32
    sampler = MinInstanceSampler(min_num_instances=4, min_size=10)

    # Expected raw and label transforms.
    raw_transform = histopathology_identity
    label_transform = PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=True,
        min_size=10,
    )

    if dataset == "nuclick":
        ds = datasets.get_nuclick_dataset(
            path=os.path.join(path, dataset),
            patch_shape=patch_shape,
            download=True,
            sampler=MinInstanceSampler(min_num_instances=2, min_size=10),
            split="Train",
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        nuclick_train_ds, nuclick_val_ds = _get_train_val_split(ds)
        if split_choice == "train":
            return nuclick_train_ds
        else:
            return nuclick_val_ds

    elif dataset == "cryonuseg":
        return datasets.get_cryonuseg_dataset(
            path=os.path.join(path, "cryonuseg"),
            patch_shape=(1, *patch_shape),
            download=True,
            sampler=sampler,
            split=split_choice,
            rater="b1",
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "pannuke":
        ds = datasets.get_pannuke_dataset(
            path=os.path.join(path, "pannuke"),
            patch_shape=(1, *patch_shape),
            download=True,
            sampler=sampler,
            ndim=2,
            folds=["fold_1", "fold_2"],
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        pannuke_train_ds, pannuke_val_ds = _get_train_val_split(ds=ds)
        if split_choice == "train":
            return pannuke_train_ds
        else:
            return pannuke_val_ds

    elif dataset == "glas":
        ds = datasets.get_glas_dataset(
            path=os.path.join(path, "glas"),
            patch_shape=patch_shape,
            download=True,
            sampler=MinInstanceSampler(min_num_instances=2),
            split="train",
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        glas_train_ds, glas_val_ds = _get_train_val_split(ds=ds)
        if split_choice == "train":
            return glas_train_ds
        else:
            return glas_val_ds

    else:
        raise NotImplementedError


def get_specialist_loaders(patch_shape, data_path, dataset):
    """This returns a selected histopathology dataset implemented in `torch_em`:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets/histopathology
    It will automatically download the dataset

    NOTE: To remove / replace the dataset with another dataset, you need to add the datasets (for train and val splits)
    in `get_specialist_datasets`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    # Get the datasets
    specialist_train_dataset = get_specialist_dataset(
        path=data_path, patch_shape=patch_shape, split_choice="train", dataset=dataset
    )
    specialist_val_dataset = get_specialist_dataset(
        path=data_path, patch_shape=patch_shape, split_choice="val", dataset=dataset
    )
    # Get the dataloaders
    train_loader = torch_em.get_data_loader(specialist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(specialist_val_dataset, batch_size=1, shuffle=True, num_workers=16)

    return train_loader, val_loader
