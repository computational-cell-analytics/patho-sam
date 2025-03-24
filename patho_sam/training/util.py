from typing import Tuple, List

import numpy as np

import torch
import torch.utils.data as data_util

from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb
from torch_em import get_data_loader

CLASS_MAP = {
    'puma': {
        2: 1,
        3: 2, 4: 2, 5: 2, 6: 2, 7: 2,
        1: 3, 8: 3,
        10: 4,
        9: 5,
    },
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
        seed: Setting a seed for your storage device for reproducibility.

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


def calculate_class_weights_for_loss_weighting(
    foreground_class_weights: List[float] = [0.4702, 0.1797, 0.2229, 0.0159, 0.1113],
) -> List[float]:
    """Calculates the class weights for weighting the cross entropy loss.

    NOTE 1: The default weights originate from weighting both the PanNuke and PUMA labels.
    TODO: Scripts coming soon!

    NOTE 2: We weigh the classes using relative integers on a scale of 1 to 10,
    where 1 resembles the most frequent class and 10 the least frequent class.

    NOTE 3: Make sure that the order of weights match the class id order.

    Args:
        foreground_class_weight: The ratio / frequency of foreground class weights.

    Returns:
        The integer weighting for each class, including the background class.
    """
    foreground_class_weights = np.array(foreground_class_weights)

    # Define the range for integer weighting.
    background_weight, max_weight = 1, 10

    # Normalize the class weights.
    min_val, max_val = np.min(foreground_class_weights), np.max(foreground_class_weights)

    # Invert the mapping (i.e. higher for rarer class, lower for common classes)
    mapped_weights = max_weight - ((foreground_class_weights - min_val) / (max_val - min_val)) * (max_weight - 1)

    # Make sure that the most common class has weight 1.
    mapped_weights[np.argmax(foreground_class_weights)] = background_weight

    # Round the weights and convert them to integer values.
    final_weights = np.round(mapped_weights).astype(int)

    # Add background weights in the beginning.
    final_weights_with_bg = [background_weight, *final_weights]

    return final_weights_with_bg


def get_sampling_weights(dataset, gamma):
    """ This class balancing approach is modified from CellViT (Hörst et al. 2024)
    """
    # The no. of patches containing the respective nuclei types over the whole dataset
    binary_weight_factors = np.array([4191, 4132, 6140, 232, 1528])
    k = np.sum(binary_weight_factors)

    # This creates a list for each patch with 1 for nucleus type present, 0 for absent for all types
    cell_counts = []
    loader = get_data_loader(dataset, batch_size=1)
    for _, label in loader:
        label = label.numpy()
        unique_classes = np.unique(label)
        cell_counts.append([int(cell_type in unique_classes) for cell_type in range(1, 6)])

    cell_counts_imgs = np.array(cell_counts)

    weight_vector = k / (gamma * binary_weight_factors + (1 - gamma) * k)  # this computes the weight per class
    img_weight = (1 - gamma) * np.max(cell_counts_imgs, axis=-1) + gamma * np.sum(
        cell_counts_imgs * weight_vector, axis=-1
    )
    img_weight[np.where(img_weight == 0)] = np.min(
        img_weight[np.nonzero(img_weight)]
    )
    return torch.Tensor(img_weight)


def get_sampler(dataset, gamma: float = 1):
    sampling_generator = torch.Generator().manual_seed(42)
    pannuke_weights = get_sampling_weights(dataset, gamma)
    sampler = data_util.WeightedRandomSampler(
        weights=pannuke_weights,
        replacement=True,
        generator=sampling_generator,
        num_samples=len(dataset),
    )
    return sampler
