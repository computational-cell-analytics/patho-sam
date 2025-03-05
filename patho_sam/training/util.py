from typing import Tuple

import numpy as np
import torch
import torch.utils.data as data_util

from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb

CLASS_MAP = {
    'puma': {
        2: 1,
        3: 2, 4: 2, 5: 2, 6: 2, 7: 2,
        1: 3, 8: 3,
        10: 4,
        9: 5,
    },
    'conic': {
        1: 2, 3: 2, 4: 2, 5: 2,
        6: 3,
        2: 5,  # this is somewhat controversial; in colon cancer, many (often most) epithelial cells are neoplastic
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


def remap_labels(y: np.ndarray, name: str) -> np.ndarray:
    """Maps the labels to overall meta classes, to match the
    semantic class structure of PanNuke dataset.

    Args:
        y: The original semantic label.
        name: The name of target dataset to remap original class ids to PanNuke class ids.

    Returns:
        The remapped labels.
    """
    if name not in CLASS_MAP:
        raise ValueError(f"The chosen dataset '{name}' is not supported.")

    # Get the class id map.
    mapping = CLASS_MAP[name]

    # Remap the labels.
    # NOTE: We go with this remapping to make sure that each ids are mapped to the exact values.
    per_id_lookup_array = np.array([mapping.get(i, 0) for i in range(max(mapping) + 1)], dtype=np.int32)
    y_remapped = per_id_lookup_array[y]
    return y_remapped
