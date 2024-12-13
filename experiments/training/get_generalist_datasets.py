import os
from typing import Optional, List

from skimage import measure

import torch
import torch.utils.data as data_util

import torch_em
from torch_em.data import datasets, MinInstanceSampler, ConcatDataset


import micro_sam.training as sam_training
from torch_em.transform.label import PerObjectDistanceTransform


"""NOTE: test sets for in-domain histopathology evaluation
    - monuseg test split
    - monusac test split
    - bcss test samples (split intrinsically - in the new PR)

length of individual loaders: @all (3 channel input images)
    - lizard:  train - 718;  val - 179
    - bcss:    train - 108;   val - 28
    - monuseg: train - 30;   val - 7
    - monusac: train - 168;   val - 41
    - pannuke: train - 1294;  val - 680
"""

def _get_train_val_split(ds, val_fraction: float = 0.2, test_exists=True):
    if not test_exists:
        ds, _ = _get_train_test_split(ds=ds)
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds

def _get_train_test_split(ds, test_fraction: float = 0.2):
    generator = torch.Generator().manual_seed(42)
    train_split, test_split = data_util.random_split(ds, [1 - test_fraction, test_fraction], generator=generator)
    return train_split, test_split


def get_concat_hp_datasets(path, patch_shape):
    label_dtype = torch.int64
    sampler = MinInstanceSampler(min_num_instances=3)
    raw_transform = sam_training.identity
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25
    )

    # datasets: CPM15, CPM17, Janowczyk, Lizard, MoNuSeg, PanNuke, PUMA, TNBC 
    cpm15_ds = datasets.get_cpm_dataset(
        path=os.path.join(path, "cpm15"), patch_shape=patch_shape, sampler=sampler, label_dtype=label_dtype,
        raw_transform=raw_transform, data_choice='cpm15', label_transform=label_transform
    )
    cpm15_train_ds, cpm15_val_ds = _get_train_val_split(ds=cpm15_ds, test_exists=False)


    cpm17_ds = datasets.get_cpm_dataset(
       path=os.path.join(path, "cpm17"), patch_shape=patch_shape, sampler=sampler, label_dtype=label_dtype,
       raw_transform=raw_transform, data_choice='cpm17', label_transform=label_transform
    )
    cpm17_train_ds, cpm17_val_ds = _get_train_val_split(ds=cpm17_ds, test_exists=False)


    janowczyk_ds = datasets.get_janowczyk_dataset(
        path=os.path.join(path, "janowczyk"), patch_shape=patch_shape, sampler=sampler, download=True, 
        label_dtype=label_dtype, raw_transform=raw_transform, annotation="nuclei", label_transform=label_transform
    )
    janowczyk_train_ds, janowczyk_val_ds = _get_train_val_split(ds=janowczyk_ds, test_exists=False)

    
    lizard_train_ds = datasets.get_lizard_dataset(
        path=os.path.join(path, "lizard"), patch_shape=patch_shape, download=True, sampler=sampler, label_dtype=label_dtype,
        split='train', label_transform=label_transform, raw_transform=raw_transform
    )
    lizard_val_ds = datasets.get_lizard_dataset(
        path=os.path.join(path, "lizard"), patch_shape=patch_shape, download=True, sampler=sampler, label_dtype=label_dtype,
        raw_transform=raw_transform, split='val', label_transform=label_transform,
    )


    monuseg_ds = datasets.get_monuseg_dataset(
        path=os.path.join(path, "monuseg"), patch_shape=patch_shape, download=True, split="train", sampler=sampler,
        label_transform=label_transform, label_dtype=label_dtype, ndim=2, raw_transform=raw_transform
    )
    monuseg_train_ds, monuseg_val_ds = _get_train_val_split(ds=monuseg_ds)
    
    
    pannuke_ds = datasets.get_pannuke_dataset(
        path=os.path.join(path, "pannuke"), patch_shape=(1, *patch_shape), download=True, sampler=MinInstanceSampler(min_num_instances=3), folds=["fold_1", "fold_2"],
        ndim=2, label_dtype=label_dtype, label_transform=label_transform, raw_transform=raw_transform
    )
    pannuke_train_ds, pannuke_val_ds = _get_train_val_split(ds=pannuke_ds)
    # pannuke_val_ds = datasets.get_pannuke_dataset(
    #     path=os.path.join(path, "pannuke"), patch_shape=(1, *patch_shape), download=True, sampler=MinInstanceSampler(min_num_instances=3), folds=["fold_2"],
    #     ndim=2, label_dtype=label_dtype, label_transform=label_transform, raw_transform=raw_transform
    # )

    

    puma_ds = datasets.get_puma_dataset(
        path=os.path.join(path, "puma"), patch_shape=patch_shape, download=True, sampler=sampler, 
        label_transform=label_transform, raw_transform=raw_transform, label_dtype=label_dtype
    )
    puma_train_ds, puma_val_ds = _get_train_val_split(ds=puma_ds, test_exists=False)


    tnbc_ds = datasets.get_tnbc_dataset(
        path=os.path.join(path, "tnbc"), patch_shape=patch_shape, download=True, sampler=sampler, 
        label_transform=label_transform, label_dtype=label_dtype, ndim=2, raw_transform=raw_transform
    )
    tnbc_train_ds, tnbc_val_ds = _get_train_val_split(tnbc_ds, test_exists=False)


    training_datasets = [
        pannuke_train_ds,
        cpm15_train_ds,
        cpm17_train_ds,
        janowczyk_train_ds,
        lizard_train_ds,
        monuseg_train_ds,
        puma_train_ds,
        tnbc_train_ds,
        pannuke_val_ds,
        cpm15_val_ds,
        cpm17_val_ds,
        janowczyk_val_ds,
        lizard_val_ds,
        monuseg_val_ds,
        puma_val_ds,
        tnbc_val_ds
    ]

    for train_dataset in training_datasets:
        print(f'{str(train_dataset)} has a length of {len(train_dataset)}')

    generalist_hp_train_dataset = ConcatDataset(
        lizard_train_ds,
        pannuke_train_ds,
        cpm15_train_ds,
        cpm17_train_ds,
        janowczyk_train_ds,
        monuseg_train_ds,
        puma_train_ds,
        # tnbc_train_ds
    )

    generalist_hp_val_dataset = ConcatDataset(
        lizard_val_ds,
        pannuke_val_ds,
        cpm15_val_ds,
        cpm17_val_ds,
        janowczyk_val_ds,
        monuseg_val_ds,
        puma_val_ds,
        # tnbc_val_ds
    )


    return generalist_hp_train_dataset, generalist_hp_val_dataset


def get_generalist_hp_loaders(patch_shape, data_path):
    """This returns the concatenated histopathology datasets implemented in `torch_em`:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets
    It will automatically download all the datasets

    NOTE: to remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    generalist_train_dataset, generalist_val_dataset = get_concat_hp_datasets(path=data_path, patch_shape=patch_shape)
    train_loader = torch_em.get_data_loader(generalist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(generalist_val_dataset, batch_size=1, shuffle=True, num_workers=16)
    return train_loader, val_loader
