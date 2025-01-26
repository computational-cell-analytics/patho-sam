import os
from glob import glob
from natsort import natsorted

import h5py
import imageio.v3 as imageio

from torch_em.data.datasets import histopathology


def get_dataset_paths(data_path, dataset) -> list:

    if dataset == "consep":
        data_paths = histopathology.consep.get_consep_paths(
            path=data_path, download=True, split="test",
        )
        label_key = 'labels'
        image_key = 'raw'

    elif dataset == "cpm15":
        image_paths, label_paths = histopathology.cpm.get_cpm_paths(
            path=data_path, download=False, split="test", data_choice="cpm15",
        )

    elif dataset == "cpm17":
        image_paths, label_paths = histopathology.cpm.get_cpm_paths(
            path=data_path, download=False, split="test", data_choice="cpm17",
        )

    elif dataset == "cryonuseg":
        image_paths, label_paths = histopathology.cryonuseg.get_cryonuseg_paths(
            path=data_path, rater_choice="b1", split="test", download=True,
        )

    elif dataset == "glas":
        data_paths = histopathology.glas.get_glas_paths(
            path=data_path, download=True, split="test",
        )
        label_key = 'labels'
        image_key = 'raw'

    elif dataset == "lizard":
        data_paths = histopathology.lizard.get_lizard_paths(
            path=data_path, download=True, split="test",
        )
        label_key = 'labels/segmentation'
        image_key = 'image'

    elif dataset == "lynsec_he":
        image_paths, label_paths = histopathology.lynsec.get_lynsec_paths(
            path=data_path, choice="h&e", download=True,
        )

    elif dataset == "lynsec_ihc":
        image_paths, label_paths = histopathology.lynsec.get_lynsec_paths(
            path=data_path, choice="ihc", download=True,
        )

    elif dataset == "monuseg":
        image_paths, label_paths = histopathology.monuseg.get_monuseg_paths(
            path=data_path, split="test", download=True,
        )

    elif dataset == "nuclick":
        image_paths, label_paths = histopathology.nuclick.get_nuclick_paths(
            path=data_path, download=True, split="Validation",
        )

    elif dataset == "nuinsseg":
        image_paths, label_paths = histopathology.nuinsseg.get_nuinsseg_paths(
            path=data_path, download=True,
        )

    elif dataset == "pannuke":
        data_paths = histopathology.pannuke.get_pannuke_paths(
            path=data_path, folds=["fold_3"], download=True,
        )
        cached_images = os.path.join(data_path, "loaded_images")
        cached_labels = os.path.join(data_path, "loaded_labels")
        os.makedirs(cached_images, exist_ok=True)
        os.makedirs(cached_labels, exist_ok=True)

        for h5_path in data_paths:
            with h5py.File(h5_path, 'r') as file:
                images = file['images']
                labels = file['labels/instances']
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

        image_paths = glob(os.path.join(cached_images, "*.tiff"))
        label_paths = glob(os.path.join(cached_labels, "*.tiff"))

    elif dataset == "puma":
        data_paths = histopathology.puma.get_puma_paths(
            path=data_path, annotations="nuclei", download=True, split="test",
        )
        label_key = 'labels/nuclei'
        image_key = 'raw'

    elif dataset == "srsanet":
        image_paths, label_paths = histopathology.srsanet.get_srsanet_paths(
            path=data_path,
            download=True,
            split="test",
        )

    elif dataset == "tnbc":
        data_paths = histopathology.tnbc.get_tnbc_paths(
            path=data_path, download=True, split="test",
        )
        label_key = 'labels/instances'
        image_key = 'raw'
        label_key = 'labels/instances'
        image_key = 'raw'

    if dataset in ["consep", "lizard", "glas", "puma", "tnbc"]:
        cached_images = os.path.join(data_path, "loaded_images")
        cached_labels = os.path.join(data_path, "loaded_labels")
        os.makedirs(cached_images, exist_ok=True)
        os.makedirs(cached_labels, exist_ok=True)
        for h5_path in data_paths:
            with h5py.File(h5_path, 'r') as file:
                img = file[image_key]
                label = file[label_key]
                img = img[:]
                label = label[:]
                image = img.transpose(1, 2, 0)

                img_path = os.path.join(
                    cached_images, os.path.basename(h5_path).replace(".h5", ".tiff"))
                label_path = os.path.join(
                    cached_labels, os.path.basename(h5_path).replace(".h5", ".tiff"))
                assert image.shape[:2] == label.shape, f"{image.shape}, {label.shape}"

                imageio.imwrite(img_path, image)
                imageio.imwrite(label_path, label)
        image_paths = glob(os.path.join(cached_images, "*.tiff"))
        label_paths = glob(os.path.join(cached_labels, "*.tiff"))

    return natsorted(image_paths), natsorted(label_paths)
