import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_util
import torch_em
from torch_em.data.datasets.histopathology import get_ignite_dataset
from torch_em.model import UNet2d
from torch_em.segmentation import default_segmentation_trainer

from patho_sam.training.util import histopathology_identity, remap_labels, remove_pad_label

ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/ignite"


def _get_train_val_split(ds, val_fraction=0.2):
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = data_util.random_split(ds, [1 - val_fraction, val_fraction], generator=generator)
    return train_ds, val_ds


def get_dataloaders(path, patch_shape):
    ignite_dataset = get_ignite_dataset(
        path,
        patch_shape,
        split="train",
        download=True,
        label_transform=partial(remap_labels, name="ignite"),
        raw_transform=histopathology_identity,
        transform=remove_pad_label,
    )
    train_ds, val_ds = _get_train_val_split(ignite_dataset)

    train_loader = torch_em.get_data_loader(train_ds, batch_size=1, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=16, shuffle=False, num_workers=16)

    return train_loader, val_loader


def visualize_loader(loader):
    import napari

    for img, label in loader:
        img = (np.squeeze(img.numpy())).transpose(1, 2, 0).astype(np.uint8)
        label = np.squeeze(label.numpy()).astype(np.uint8)
        viewer = napari.Viewer()
        viewer.add_image(img, name="img")
        viewer.add_labels(label, name="label")
        napari.run()


def train_foreground_segmentation_model(args):
    train_loader, val_loader = get_dataloaders(args.input_path, patch_shape=(512, 512))

    visualize_loader(train_loader)

    unet = UNet2d(in_channels=3, out_channels=2, depth=4, initial_features=64, gain=2, final_activation=None)

    trainer = default_segmentation_trainer(
        "foreground_sep",
        model=unet,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=nn.CrossEntropyLoss(),
        save_root=args.save_root,
    )

    trainer.fit(args.n_iterations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, default=ROOT)
    parser.add_argument("--save_root", "-s", type=str)
    parser.add_argument("--n_iterations")
    args = parser.parse_args()
    train_foreground_segmentation_model(args)


if __name__ == "__main__":
    main()
