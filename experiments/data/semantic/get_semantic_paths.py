import os
from glob import glob
from natsort import natsorted

import h5py
import imageio.v3 as imageio

from torch_em.data.datasets import histopathology

from patho_sam.training.util import remap_labels


def get_dataset_paths(data_path, dataset):
    cached_images = os.path.join(data_path, "loaded_images")
    cached_labels = os.path.join(data_path, "loaded_labels")
    os.makedirs(cached_images, exist_ok=True)
    os.makedirs(cached_labels, exist_ok=True)

    if dataset == "pannuke":
        data_paths = histopathology.pannuke.get_pannuke_paths(
            path=data_path, folds=["fold_3"], download=True,
        )

        for h5_path in data_paths:
            with h5py.File(h5_path, 'r') as file:
                images = file['images']
                labels = file['labels/semantic']
                images = images[:]
                labels = labels[:]

                # PanNuke is provided in an array of shape (C, B, H, W)
                images = images.transpose(1, 2, 3, 0)  # --> (B, H, W, C)

                counter = 1
                for image, label in zip(images, labels):
                    image_path = os.path.join(cached_images, f"{counter:04}.tiff")
                    label_path = os.path.join(cached_labels, f"{counter:04}.tiff")

                    assert image.shape == (256, 256, 3)
                    imageio.imwrite(image_path, image)
                    imageio.imwrite(label_path, label)

                    counter += 1

    elif dataset == "puma":
        data_paths = histopathology.puma.get_puma_paths(
            path=data_path, annotations="nuclei", download=True, split="test",
        )

        for h5_path in data_paths:
            with h5py.File(h5_path, 'r') as file:
                img = file['raw']
                label = file['labels/semantic/nuclei']
                img = img[:]
                label = label[:]
                image = img.transpose(1, 2, 0)
                label = remap_labels(label, name=None)
                img_path = os.path.join(
                    cached_images, os.path.basename(h5_path).replace(".h5", ".tiff"))
                label_path = os.path.join(
                    cached_labels, os.path.basename(h5_path).replace(".h5", ".tiff"))
                assert image.shape[:2] == label.shape, f"{image.shape}, {label.shape}"

                imageio.imwrite(img_path, image)
                imageio.imwrite(label_path, label)

    else:
        raise ValueError

    image_paths = glob(os.path.join(cached_images, "*.tiff"))
    label_paths = glob(os.path.join(cached_labels, "*.tiff"))

    return natsorted(image_paths), natsorted(label_paths)
