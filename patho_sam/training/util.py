from typing import Tuple, List, Callable
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data_util

from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb
import kornia.augmentation as K


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
    foreground_class_weights: List[float] = [0.507, 0.1082, 0.2284, 0.0038, 0.1526],
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
    background_weight, max_weight = 1, 3

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


def get_sampling_weights(instance_dataset, semantic_dataset, gamma: float, input_path, split):

    # If weights for the split have already been extracted and saved, they are loaded
    weights_csv_path = os.path.join(input_path, f"{split}_instance_sampling_weights.csv")

    if os.path.exists(weights_csv_path):
        print(f"Sampling weights for the {split} set have already been extracted.")
        df = pd.read_csv(weights_csv_path)
        cell_type_presence = df.to_numpy()

    # This creates an array where each line represents a training sample and each column corresponds to the binary
    # presence of each nucleus type (1, 2, 3, 4, 5)

    # Class-Pixel-level
    # else:
    #     cell_type_presence = np.array(
    #         [[torch.sum(label == cell_type).item() for cell_type in range(1, 6)]
    #             for _, label in tqdm(dataset, desc="Extracting sampling weights")])

    #     df = pd.DataFrame(cell_type_presence)
    #     df.to_csv(weights_csv_path, index=False)

    # Class-Instance-level
    else:
        cell_type_presence = np.array(
            [[len(np.unique(instance_label[semantic_label == cell_type]))
              for cell_type in range(1, 6)]for (_, instance_label), (_, semantic_label) in
                tqdm(zip(instance_dataset, semantic_dataset), total=len(instance_dataset))]
            )

        df = pd.DataFrame(cell_type_presence)
        df.to_csv(weights_csv_path, index=False)

    binary_weight_factors = np.sum(cell_type_presence, axis=0)

    k = np.sum(binary_weight_factors)

    # This creates an array with the respective weight factor for each nucleus type
    weight_vector = k / (gamma * binary_weight_factors + (1 - gamma) * k)

    # This applies the weight factor to all the training samples with respect to the set gamma value
    img_weight = (1 - gamma) * np.max(cell_type_presence, axis=-1) + gamma * np.sum(
        (cell_type_presence * weight_vector), axis=1)

    # This assigns the minimal non-zero sample weight to samples whose weight is 0
    img_weight[np.where(img_weight == 0)] = np.min(
        img_weight[np.nonzero(img_weight)]
    )

    return torch.Tensor(img_weight)


def get_sampler(instance_dataset, semantic_dataset, gamma, path, split) -> data_util.Sampler:
    pannuke_weights = get_sampling_weights(instance_dataset, semantic_dataset, gamma, path, split)

    sampler = data_util.WeightedRandomSampler(
        weights=pannuke_weights,
        replacement=True,
        num_samples=len(instance_dataset),
    )

    return sampler


def geometric_transforms(x, y, seq):
    x = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(0)
    y = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    x, y = seq(x, y)
    x = (x.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return x, y.squeeze(0).numpy()


def photometric_transforms(x, seq):
    x = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(0)
    x = seq(x)
    x = (x.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return x


def build_transforms(patch_shape) -> Tuple[Callable, Callable]:
    geometric_transforms_list = [
        K.RandomRotation90(p=0.5, times=(1, 2)),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomResizedCrop(size=(patch_shape[1], patch_shape[2]),
                            scale=(0.5, 0.5), p=0.15),
    ]
    photometric_transforms_list = [
        K.RandomGaussianBlur(kernel_size=(11, 11), sigma=(0.5, 2.0), p=0.2),
        K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.25),
        K.ColorJitter(brightness=0.25, contrast=0.25,
                      saturation=0.1, hue=0.05, p=0.2),
    ]

    geometric_seq = K.AugmentationSequential(*geometric_transforms_list, data_keys=["input", "mask"])
    photometric_seq = K.AugmentationSequential(*photometric_transforms_list, data_keys=["input"])

    return geometric_seq, photometric_seq

# def get_transforms(patch_shape) -> Tuple[Callable, Callable]:
#     import kornia.augmentation as K
#     transform_settings = {
#         "randomrotate90": {"p": 0.5},
#         "horizontalflip": {"p": 0.5},
#         "verticalflip": {"p": 0.5},
#         "downscale": {"p": 0.15, "scale": 0.5},  # scale as fraction of original size
#         "blur": {"p": 0.2, "kernel_size": 11},  # kernel_size must be odd
#         "gaussnoise": {"p": 0.25, "std": 0.05},
#         "colorjitter": {
#             "p": 0.2,
#             "brightness": 0.25,
#             "contrast": 0.25,
#             "saturation": 0.1,
#             "hue": 0.05
#         },
#         # "normalize": {
#         #     "mean": [0.5, 0.5, 0.5],
#         #     "std": [0.5, 0.5, 0.5]
#         # }
#     }

#     geometric_transforms_list = []

#     photometric_transforms_list = []

#     # Random 90° rotation
#     geometric_transforms_list.append(K.RandomRotation90(p=transform_settings["randomrotate90"]["p"], times=(1, 2)))

#     # Horizontal flip
#     geometric_transforms_list.append(K.RandomHorizontalFlip(p=transform_settings["horizontalflip"]["p"]))

#     # Vertical flip
#     geometric_transforms_list.append(K.RandomVerticalFlip(p=transform_settings["verticalflip"]["p"]))

#     # Downscale (simulated via RandomResizedCrop)
#     geometric_transforms_list.append(
#         K.RandomResizedCrop(
#             size=(patch_shape[1], patch_shape[2]),
#             scale=(transform_settings["downscale"]["scale"], transform_settings["downscale"]["scale"]),
#             p=transform_settings["downscale"]["p"]
#         )
#     )

#     # Blur
#     photometric_transforms_list.append(
#         K.RandomGaussianBlur(
#             kernel_size=(transform_settings["blur"]["kernel_size"], transform_settings["blur"]["kernel_size"]),
#             sigma=(0.1, 2.0),
#             p=transform_settings["blur"]["p"]
#         )
#     )

#     # Gaussian noise
#     photometric_transforms_list.append(
#         K.RandomGaussianNoise(
#             mean=0.0,
#             std=transform_settings["gaussnoise"]["std"],
#             p=transform_settings["gaussnoise"]["p"]
#             )
#     )

#     # Color jitter
#     photometric_transforms_list.append(
#         K.ColorJitter(
#             brightness=transform_settings["colorjitter"]["brightness"],
#             contrast=transform_settings["colorjitter"]["contrast"],
#             saturation=transform_settings["colorjitter"]["saturation"],
#             hue=transform_settings["colorjitter"]["hue"],
#             p=transform_settings["colorjitter"]["p"]
#         )
#     )
#     # Normalize
#     # mean = torch.tensor(transform_settings.get("normalize", {}).get("mean", [0.5, 0.5, 0.5]))
#     # std = torch.tensor(transform_settings.get("normalize", {}).get("std", [0.5, 0.5, 0.5]))
#     # transform_list.append(K.Normalize(mean=mean, std=std))

#     # Compose
#     def geometric_transforms(x, y):
#         x = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(0)
#         y = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
#         x, y = K.AugmentationSequential(*geometric_transforms_list, data_keys=["input", "mask"])(x, y)
#         x = (x.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)  # back to original type
#         return x, y.squeeze(0).numpy()

#     def photometric_transforms(x):
#         x = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(0)
#         x = K.AugmentationSequential(*photometric_transforms_list, data_keys=["input"])(x)
#         x = x.squeeze(0).numpy()
#         x = (x * 255).clip(0, 255).astype(np.uint8)  # back to original type
#         return x

#     return geometric_transforms, photometric_transforms


def get_sampling_weights_cellvit(dataset, gamma: float):
    """ This class balancing approach is modified from CellViT (Hörst et al. 2024)
    """

    # This creates an array where each line represents a training sample and each column corresponds to the binary
    # presence of each nucleus type (1, 2, 3, 4, 5)
    cell_type_presence = np.array(
        [[int(cell_type in np.unique(label)) for cell_type in range(1, 6)] for _, label in dataset])

    # We create an array of the number of samples that each nucleus type is represented in
    binary_weight_factors = np.sum(cell_type_presence, axis=0)

    k = np.sum(binary_weight_factors)

    # This creates an array with the respective weight factor for each nucleus type
    weight_vector = k / (gamma * binary_weight_factors + (1 - gamma) * k)

    # This applies the weight factor to all the training samples with respect to the set gamma value
    img_weight = (1 - gamma) * np.max(cell_type_presence, axis=-1) + gamma * np.sum(
        cell_type_presence * weight_vector, axis=-1
    )

    # This assigns the minimal non-zero sample weight to samples whose weight is 0
    img_weight[np.where(img_weight == 0)] = np.min(
        img_weight[np.nonzero(img_weight)]
    )

    return torch.Tensor(img_weight)

# class DeterministicDataset(torch.utils.data.Dataset):
#     def __init__(self, base_dataset):
#         self.base_dataset = base_dataset

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         # Access original dataset item without applying augmentations
#         # Assumes your base dataset has `get_raw_item(idx)` or similar
#         # If not, you can temporarily disable transforms
#         item = self.base_dataset.get_raw_item(idx)  
#         return item
    
class DeterministicSubset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.base_dataset = subset.dataset
        self.indices = subset.indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        orig_transform = self.base_dataset.transform
        self.base_dataset.transform = None
        item = self.base_dataset[real_idx]
        self.base_dataset.transform = orig_transform
        return item

class DeterministicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Resolve Subset / ConcatDataset
        ds = self.dataset
        while hasattr(ds, 'dataset'):
            if isinstance(ds, torch.utils.data.Subset):
                idx = ds.indices[idx]
            ds = ds.dataset

        if hasattr(ds, 'transform'):
            orig_transform = ds.transform
            ds.transform = None

        item = ds[idx]

        if hasattr(ds, 'transform'):
            ds.transform = orig_transform

        return item