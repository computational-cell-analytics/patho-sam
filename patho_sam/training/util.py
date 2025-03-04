from typing import Tuple

import numpy as np
import torch
import torch.utils.data as data_util

from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb

CLASS_MAP = {
    'puma': {
        1: 3,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 3,
        9: 5,
        10: 4,
    },
    'conic': {
        1: 2,
        2: 5,  # this is somewhat controversial; in colon cancer, many (often most) epithelial cells are neoplastic 
        3: 2,
        4: 2,
        5: 2,
        6: 3,
    }
}

CLASS_DICT = {
    'puma': {
        "nuclei_stroma": 1,
        "nuclei_tumor": 2,
        "nuclei_plasma_cell": 3,
        "nuclei_histiocyte": 4,
        "nuclei_lymphocyte": 5,
        "nuclei_melanophage": 6,
        "nuclei_neutrophil": 7,
        "nuclei_endothelium": 8,
        "nuclei_epithelium": 9,
        "nuclei_apoptosis": 10
    },
    'pannuke': {
        "neoplastic": 1,
        "inflammatory": 2,
        "connective / soft tissue": 3,
        "dead cells": 4,
        "epithelial": 5,
    },
    'conic': {
        "neutrophil": 1,
        "epithelial": 2,
        "lymphocyte": 3,
        "plasma": 4,
        "eosinophil": 5,
        "connective": 6,
    }
}


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


def get_train_val_split(
    ds: torch.utils.data.Dataset, val_fraction: float = 0.2, seed: int = 42,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Creates split for a dataset for a decided fraction.

    Args:
        dataset: The segmentation dataset.
        val_fraction: The fraction of split to decide for validation, and remanining for test.

    Returns:
        Tuple of train and val datasets.
    """
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds


def remap_puma(y):
    max_key = max(CLASS_MAP['puma'].keys())
    lookup_array = np.zeros(max_key + 1, dtype=np.int32)
    for old_id, new_id in CLASS_MAP['puma'].items():
        lookup_array[old_id] = new_id
    remapped_label = np.take(lookup_array, y)
    return remapped_label


def remap_conic(y):
    max_key = max(CLASS_MAP['conic'].keys())
    lookup_array = np.zeros(max_key + 1, dtype=np.int32)
    for old_id, new_id in CLASS_MAP['conic'].items():
        lookup_array[old_id] = new_id
    remapped_label = np.take(lookup_array, y)
    return remapped_label
