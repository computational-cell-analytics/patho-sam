import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Union, Tuple
from pathlib import Path

import json
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
import torch_em
from torch_em.data.datasets.util import split_kwargs


def _get_data_split(path, split) -> List:
    json_path = os.path.join(path, 'ignite_ihc_split.json')
    if os.path.exists(json_path):
        print(f"Found existing split file at '{json_path}'.")
        with open(json_path, 'r') as f:
            split_dict = json.load(f)

    else:
        print(f"Creating a new split file at '{json_path}'.")
        image_names = natsorted([Path(img).stem for img in glob(os.path.join(path, "ignite_ihc", "*.h5"))])
        train_ids, test_ids = train_test_split(image_names, test_size=0.2, random_state=42)  # 20% split for test.
        train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)  # 15% split for val.
        split_dict = {"train": train_ids, "val": val_ids, "test": test_ids}
        with open(json_path, 'w') as f:
            json.dump(split_dict, f, indent=2)

    return split_dict[split]


def get_ignite_paths(path, split) -> List:
    split_list = _get_data_split(path, split)
    volume_paths = [os.path.join(path, "ignite_ihc", f"{img_name}.h5") for img_name in split_list]
    return volume_paths


def get_ignite_dataset(path, split, patch_shape, **kwargs) -> Dataset:
    volume_paths = get_ignite_paths(path, split)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="image",
        label_paths=volume_paths,
        label_key="instance_pred",
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_ignite_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    **kwargs
) -> DataLoader:
    """Get the ignite dataloader for nuclei segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ignite_dataset(
        path, split, patch_shape, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
