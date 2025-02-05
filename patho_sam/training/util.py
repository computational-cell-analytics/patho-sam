from typing import Tuple

import torch
import torch.utils.data as data_util

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
