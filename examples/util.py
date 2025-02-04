import os
from glob import glob
from natsort import natsorted


import torch
import torch.utils.data as data_util
import torch_em
from torch_em.data import MinInstanceSampler, datasets
from torch_em.transform.label import PerObjectDistanceTransform
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


def _get_train_val_split(ds, val_fraction=0.2):
    """Dataset split. This splits the provided dataset used for training into a train and val set.
    """
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds


def get_dataloaders(patch_shape, batch_size, data_path, dataset=None, images_dir=None, masks_dir=None):
    label_dtype = torch.float32
    sampler = MinInstanceSampler(min_num_instances=4, min_size=10)

    # Expected raw and label transforms for training
    raw_transform = histopathology_identity
    label_transform = PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=True,
        min_size=10,
    )

    ds = None

    if dataset == "consep":
        ds = datasets.get_consep_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            split="train",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "cpm15":
        train_ds = datasets.get_cpm_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=False,
            split="train",
            data_choice="cpm15",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        val_ds = datasets.get_cpm_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=False,
            split="val",
            data_choice="cpm15",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "cpm17":
        ds = datasets.get_cpm_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=False,
            split="train",
            data_choice="cpm17",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "cryonuseg":
        train_loader = datasets.get_cryonuseg_loader(
            path=data_path,
            patch_shape=(1,) + patch_shape,
            rater="b1",
            split="train",
            download=True,
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        val_loader = datasets.get_cryonuseg_loader(
            path=data_path,
            patch_shape=(1,) + patch_shape,
            rater="b1",
            split="val",
            download=True,
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "glas":
        ds = datasets.get_glas_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            split="train",
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
            sampler=MinInstanceSampler(min_num_instances=2),
        )

    elif dataset == "lizard":
        train_ds = datasets.get_lizard_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            split="train",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        val_ds = datasets.get_lizard_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            split="val",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    # lynsec_he will not have an exclusive test split but the whole dataset will be used for training
    elif dataset == "lynsec_he":
        ds = datasets.get_lynsec_dataset(
            path=data_path,
            patch_shape=patch_shape,
            choice="h&e",
            download=True,
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    # lynsec_ihc will not have an exclusive test split but the whole dataset will be used for training
    elif dataset == "lynsec_ihc":
        ds = datasets.get_lynsec_dataset(
            path=data_path,
            patch_shape=patch_shape,
            choice="ihc",
            download=True,
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "monusac":
        ds = datasets.get_monusac_dataset(
            path=data_path,
            patch_shape=patch_shape,
            split="train",
            download=True,
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "monuseg":
        ds = datasets.get_monuseg_dataset(
            path=data_path,
            patch_shape=patch_shape,
            split="train",
            download=True,
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    # nuinsseg will not have an exclusive test split but the whole dataset will be used for training
    elif dataset == "nuinsseg":
        ds = datasets.get_nuinsseg_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "nuclick":
        ds = datasets.get_nuclick_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            split="Train",
            sampler=MinInstanceSampler(min_num_instances=2, min_size=10),
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "pannuke":
        ds = datasets.get_pannuke_dataset(
            path=data_path,
            patch_shape=(1,) + patch_shape,
            folds=["fold_1", "fold_2"],
            ndim=2,
            download=True,
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "puma":
        train_ds = datasets.get_puma_dataset(
            path=data_path,
            patch_shape=patch_shape,
            annotations="nuclei",
            download=True,
            split="train",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        val_ds = datasets.get_puma_dataset(
            path=data_path,
            patch_shape=patch_shape,
            batch_size=1,
            annotations="nuclei",
            download=True,
            split="val",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "srsanet":
        train_ds = datasets.get_srsanet_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            split="train",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        val_ds = datasets.get_srsanet_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            split="val",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset == "tnbc":
        train_ds = datasets.get_tnbc_dataset(
            path=data_path,
            patch_shape=patch_shape,
            ndim=2,
            download=True,
            split="train",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )
        val_ds = datasets.get_tnbc_dataset(
            path=data_path,
            patch_shape=patch_shape,
            ndim=2,
            download=True,
            split="val",
            sampler=sampler,
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
        )

    elif dataset is None:
        raw_paths = natsorted(glob(os.path.join(images_dir, "*")))
        label_paths = natsorted(glob(os.path.join(masks_dir, "*")))

        assert len(raw_paths) == len(label_paths)

        ds = torch_em.default_segmentation_dataset(
            raw_paths=raw_paths,
            raw_key=None,
            label_paths=label_paths,
            label_key=None,
            patch_shape=patch_shape,
            ndim=2,
            with_channels=True,
            sampler=MinInstanceSampler(min_num_instances=2),
            label_dtype=label_dtype,
            label_transform=label_transform,
            raw_transform=raw_transform,
            )

    else:
        raise NotImplementedError

    if ds is not None:
        train_ds, val_ds = _get_train_val_split(ds=ds)

    train_loader = torch_em.get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=1, shuffle=True, num_workers=16)

    return train_loader, val_loader
