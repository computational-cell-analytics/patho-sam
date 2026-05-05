import os
from pathlib import Path
from typing import List, Literal, Tuple, Union

import torch_em
from torch.utils.data import DataLoader, Dataset
from torch_em.data.datasets.util import split_kwargs


def get_cancerscout_paths(path, split: str, entities: List) -> List:
    path = Path(path)
    volume_paths = []

    for entity in entities:
        volume_paths.extend(sorted((path / f"{split}_models" / f"new_{entity}_data" / "fixed_h5_files").glob("*.h5")))

    return [str(p) for p in volume_paths]


def get_cancerscout_dataset(
    path, split, patch_shape, entities: Literal["tumor", "non_tumor"] = ["tumor", "non_tumor"], **kwargs
) -> Dataset:
    volume_paths = get_cancerscout_paths(path, split, entities)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="img",
        label_paths=volume_paths,
        label_key="inst_labels/v_2",
        patch_shape=patch_shape,
        with_channels=True,
        is_seg_dataset=True,
        ndim=2,
        **kwargs,
    )


def get_cancerscout_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "eval"],
    entities: Literal["tumor", "non_tumor"] = ["tumor", "non_tumor"],
    **kwargs,
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
    dataset = get_cancerscout_dataset(path, split, patch_shape, entities, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
