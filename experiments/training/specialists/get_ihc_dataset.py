import os

import torch

import torch_em
from torch_em.data import MinInstanceSampler, datasets, ConcatDataset
from torch_em.transform.label import PerObjectDistanceTransform

from patho_sam.training import histopathology_identity


def get_specialist_datasets(path, patch_shape, split_choice):
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

    lynsec_ds = datasets.get_lynsec_dataset(
        path=os.path.join(path, 'lynsec'),
        patch_shape=patch_shape,
        download=True,
        sampler=sampler,
        split=split_choice,
        choice='ihc',
        label_dtype=label_dtype,
        label_transform=label_transform,
        raw_transform=raw_transform,
    )

    srsanet_ds = datasets.get_srsanet_dataset(
        path=os.path.join(path, "srsanet"),
        patch_shape=patch_shape,
        split=split_choice,
        download=True,
        sampler=sampler,
        label_dtype=label_dtype,
        label_transform=label_transform,
        raw_transform=raw_transform,
    )

    _datasets = [
       lynsec_ds, srsanet_ds
    ]

    return ConcatDataset(*_datasets)


def get_specialist_loaders(patch_shape, data_path):
    """This returns a selected histopathology dataset implemented in `torch_em`:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets/histopathology
    It will automatically download the dataset

    NOTE: To remove / replace the dataset with another dataset, you need to add the datasets (for train and val splits)
    in `get_specialist_datasets`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    # Get the datasets
    specialist_train_dataset = get_specialist_datasets(
        path=data_path, patch_shape=patch_shape, split_choice="train",
    )
    specialist_val_dataset = get_specialist_datasets(
        path=data_path, patch_shape=patch_shape, split_choice="val",
    )
    # Get the dataloaders
    train_loader = torch_em.get_data_loader(specialist_train_dataset, batch_size=2, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(specialist_val_dataset, batch_size=1, shuffle=True, num_workers=16)

    return train_loader, val_loader
