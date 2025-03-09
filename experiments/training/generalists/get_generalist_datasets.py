import os

import torch
import torch.utils.data as data_util

import torch_em
from torch_em.data import ConcatDataset, MinInstanceSampler, datasets
from torch_em.transform.label import PerObjectDistanceTransform

from patho_sam.training import histopathology_identity


def _get_train_val_split(ds, val_fraction=0.2):
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds


def get_concat_hp_datasets(path, patch_shape, split_choice):
    # Important stuff for dataloaders.
    label_dtype = torch.float32
    sampler = MinInstanceSampler(min_num_instances=4, min_size=10)

    # Expected raw and label transforms.
    raw_transform = histopathology_identity
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=10,
    )

    # Datasets used for training: CPM15, CPM17, Lizard, MoNuSeg, PanNuke, PUMA, TNBC
    cpm15_ds = datasets.get_cpm_dataset(
        path=os.path.join(path, "cpm15"), patch_shape=patch_shape, sampler=sampler,
        label_dtype=label_dtype, raw_transform=raw_transform, data_choice="cpm15",
        split=split_choice, label_transform=label_transform, n_samples=50,  # NOTE: oversampling the data.
    )

    lizard_ds = datasets.get_lizard_dataset(
        path=os.path.join(path, "lizard"), patch_shape=patch_shape, download=True, sampler=sampler,
        label_dtype=label_dtype, split=split_choice, label_transform=label_transform, raw_transform=raw_transform,
    )

    puma_ds = datasets.get_puma_dataset(
        path=os.path.join(path, "puma"), patch_shape=patch_shape, download=True, sampler=sampler, split=split_choice,
        label_transform=label_transform, raw_transform=raw_transform, label_dtype=label_dtype,
    )

    tnbc_ds = datasets.get_tnbc_dataset(
        path=os.path.join(path, "tnbc"), patch_shape=patch_shape, download=True, sampler=sampler,
        split=split_choice, label_transform=label_transform, label_dtype=label_dtype,
        ndim=2, raw_transform=raw_transform, n_samples=50,  # NOTE: oversampling the data.
    )

    def _get_cpm17_dataset():
        cpm17_ds = datasets.get_cpm_dataset(
            path=os.path.join(path, "cpm17"), patch_shape=patch_shape, sampler=sampler,
            label_dtype=label_dtype, raw_transform=raw_transform, data_choice="cpm17",
            split="train", label_transform=label_transform, n_samples=50,  # NOTE: oversampling the data.
        )
        cpm17_train_ds, cpm17_val_ds = _get_train_val_split(ds=cpm17_ds)
        if split_choice == "train":
            return cpm17_train_ds
        else:
            return cpm17_val_ds

    def _get_monuseg_dataset():
        monuseg_ds = datasets.get_monuseg_dataset(
            path=os.path.join(path, "monuseg"), patch_shape=patch_shape, download=True, split="train",
            sampler=sampler, label_transform=label_transform, label_dtype=label_dtype,
            ndim=2, raw_transform=raw_transform, n_samples=50,  # NOTE: oversampling the data.
        )
        monuseg_train_ds, monuseg_val_ds = _get_train_val_split(ds=monuseg_ds)
        if split_choice == "train":
            return monuseg_train_ds
        else:
            return monuseg_val_ds

    def _get_pannuke_dataset():
        pannuke_ds = datasets.get_pannuke_dataset(
            path=os.path.join(path, "pannuke"), patch_shape=(1, *patch_shape), download=True,
            sampler=sampler, ndim=2, folds=["fold_1", "fold_2"], label_dtype=label_dtype,
            label_transform=label_transform, raw_transform=raw_transform,
        )
        pannuke_train_ds, pannuke_val_ds = _get_train_val_split(ds=pannuke_ds)
        if split_choice == "train":
            return pannuke_train_ds
        else:
            return pannuke_val_ds

    _datasets = [
        cpm15_ds, lizard_ds, puma_ds, tnbc_ds, _get_cpm17_dataset(), _get_monuseg_dataset(), _get_pannuke_dataset()
    ]

    return ConcatDataset(*_datasets)


def get_generalist_hp_loaders(patch_shape, data_path):
    """This returns the concatenated histopathology datasets implemented in `torch_em`:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets/histopathology
    It will automatically download all the datasets

    NOTE: To remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_hp_datasets`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    # Get the datasets
    generalist_train_dataset = get_concat_hp_datasets(path=data_path, patch_shape=patch_shape, split_choice="train")
    generalist_val_dataset = get_concat_hp_datasets(path=data_path, patch_shape=patch_shape, split_choice="val")
    # Get the dataloaders
    train_loader = torch_em.get_data_loader(generalist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(generalist_val_dataset, batch_size=1, shuffle=True, num_workers=16)

    return train_loader, val_loader
