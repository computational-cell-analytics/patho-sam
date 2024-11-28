"""The TNBC dataset contains annotations for nucleus segmentation
in H&E stained histopathology images.

The dataset is located at https://doi.org/10.5281/zenodo.1175282.
Please cite it if you use this dataset for your research.
"""
import numpy as np
import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em
import h5py
from torch_em.data.datasets import util
import tifffile


URL = "https://zenodo.org/records/1175282/files/TNBC_NucleiSegmentation.zip"
CHECKSUM = "da708c3a988f4ad4b9bbb9283b387faf703f0bc0e5e689927306bd27ea13a57f"


def _preprocess_images(path):
    import h5py

    raw_paths = natsorted(glob(os.path.join(path, "TNBC_NucleiSegmentation", "Slide_*", "*.png")))
    label_paths = natsorted(glob(os.path.join(path, "TNBC_NucleiSegmentation", "GT_*", "*.png")))

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for rpath, lpath in tqdm(zip(raw_paths, label_paths), desc="Preprocessing images", total=len(raw_paths)):
        raw = imageio.imread(rpath)
        raw = raw[..., :-1].transpose(2, 0, 1)  # remove 4th alpha channel (seems like an empty channel).
        label = imageio.imread(lpath)
        vol_path = os.path.join(preprocessed_dir, f"{Path(lpath).stem}.h5")

        with h5py.File(vol_path, "w") as f:
            f.create_dataset("raw", shape=raw.shape, data=raw, compression="gzip")
            f.create_dataset("labels/semantic", shape=label.shape, data=label, compression="gzip")
            f.create_dataset(
                "labels/instances", shape=label.shape, data=connected_components(label), compression="gzip"
            )

    shutil.rmtree(os.path.join(path, "TNBC_NucleiSegmentation"))
    shutil.rmtree(os.path.join(path, "__MACOSX"))


def get_tiffs(path):
    output_dir = os.path.join(path)
    os.makedirs((os.path.join(output_dir, 'images')), exist_ok=True)
    os.makedirs((os.path.join(output_dir, 'labels')), exist_ok=True)
    for file in glob(os.path.join(path, 'preprocessed', '*.h5')): 
        with h5py.File(file, 'r') as f: 
            img_data = f['raw']
            label_data = f['labels/instances']
            basename = os.path.basename(file)
            name, ext = os.path.splitext(basename)
            img_output_path = os.path.join(output_dir, 'images', f'{name}.tiff')
            tifffile.imwrite(img_output_path, img_data)
            label_output_path = os.path.join(output_dir, 'labels', f'{name}.tiff')
            tifffile.imwrite(label_output_path, label_data)
    image_paths = natsorted(glob(os.path.join(path, 'images', '*.tiff')))
    label_paths = natsorted(glob(os.path.join(path, 'labels', '*.tiff')))
    return image_paths, label_paths


def get_tnbc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TNBC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded data.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "TNBC_NucleiSegmentation.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    _preprocess_images(path)


def get_tnbc_paths(path: Union[os.PathLike, str], download: bool = False) -> List[int]:
    """Get paths to the TNBC data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed image data.
    """
    get_tnbc_data(path, download)
    image_paths, label_paths = get_tiffs(path)
    return image_paths, label_paths


def get_tnbc_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the TNBC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_tnbc_paths(path, download)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=False, boundaries=False, offsets=None
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_tnbc_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> DataLoader:
    """Get the TNBC dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tnbc_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
