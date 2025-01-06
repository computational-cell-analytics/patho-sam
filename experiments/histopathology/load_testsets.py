import os
from glob import glob

import numpy as np
from dataloaders import get_dataloaders
from natsort import natsorted
from skimage import io
from util import dataloading_args

DATASETS = [
    "bcss",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "janowczyk",
    "lizard",
    "lynsec_he",
    "lynsec_ihc",
    "monusac",
    "monuseg",
    "nuclick",
    "nuinsseg",
    "pannuke",
    "puma",
    "srsanet",
    "tnbc",
]


def remove_empty_labels(path):
    empty_count = 0
    file_list = natsorted(glob(os.path.join(path, "labels", "*.tiff")))
    for image_path in file_list:
        img = io.imread(image_path)
        unique_elements = np.unique(img)
        if len(unique_elements) == 1:
            print(f"Image {os.path.basename(image_path)} does not contain labels and will be removed.")
            empty_count += 1
            os.remove(image_path)
            os.remove(os.path.join(path, "images", f"{os.path.basename(image_path)}"))
            assert len(os.listdir(os.path.join(path, "labels"))) == len(os.listdir(os.path.join(path, "images")))
    print(f"{empty_count} labels were empty")
    label_len = len(os.listdir(os.path.join(path, "labels")))
    print(f"There are {label_len} images left")


def load_testsets(path, dsets=DATASETS, patch_shape=(512, 512)):
    for dataset in dsets:
        if os.path.exists(os.path.join(path, dataset, "loaded_testset", "images")):
            if len(os.listdir(os.path.join(path, dataset, "loaded_testset", "images"))) > 1:
                print(f"Dataset {dataset} is loaded already.")
                continue
        counter = 1
        print(f"Loading {dataset} dataset...")
        dpath = os.path.join(path, dataset)
        os.makedirs(dpath, exist_ok=True)
        loaders = []
        if dataset in [
            "cryonuseg",
            "lynsec_he",
            "lynsec_ihc",
            "nuinsseg",
        ]:  # out of domain datasets, loads entire datasets
            loaders.append(get_dataloaders(patch_shape, dpath, dataset))
        elif dataset in [  # in-domain datasets, loads only test split
            "cpm15",
            "cpm17",
            "lizard",
            "monuseg",
            "puma",
            "tnbc",
        ]:
            loaders.append(get_dataloaders(patch_shape, dpath, dataset, split="test"))
        elif dataset == "monusac":  # out of donain, partial dataset
            for split in ["train", "test"]:
                loaders.append(get_dataloaders(patch_shape, dpath, dataset, split=split))
        elif dataset == "janowczyk":
            for split in ["train"]:
                loaders.append(get_dataloaders(patch_shape, dpath, dataset, split=split))
        elif dataset == "pannuke":
            loaders.append(get_dataloaders(patch_shape, dpath, dataset, split=["fold_3"]))
        image_output_path = os.path.join(path, dataset, "loaded_testset", "images")
        label_output_path = os.path.join(path, dataset, "loaded_testset", "labels")
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)
        for loader in loaders:
            for image, label in loader:
                image_array = image.numpy()
                label_array = label.numpy()
                squeezed_image = image_array.squeeze()
                label_data = label_array.squeeze()
                tp_img = squeezed_image.transpose(1, 2, 0)
                if tp_img.shape[-1] == 4:  # deletes alpha channel if one exists
                    tp_img = tp_img[..., :-1]
                # 3 tnbc images had a shape of (512, 512, 2) and had to be sorted out
                if tp_img.shape != (patch_shape[0], patch_shape[1], 3):
                    print(
                        f"Incorrect image shape of {tp_img.shape} in "
                        f"{os.path.join(image_output_path, f'{counter:04}.tiff')}"
                    )
                    counter += 1
                    continue
                io.imwrite(os.path.join(image_output_path, f"{counter:04}.tiff"), tp_img)
                io.imwrite(os.path.join(label_output_path, f"{counter:04}.tiff"), label_data)
                counter += 1
            remove_empty_labels(os.path.join(path, dataset, "loaded_testset"))
            print(f"{dataset} testset has successfully been loaded.")


def main():
    args = dataloading_args()
    if args.path is not None:
        data_path = args.path
    else:
        data_path = "/mnt/lustre-grete/usr/u12649/scratch/data/final_test/"
    if args.datasets is not None:
        load_testsets(data_path, [args.datasets], args.patch_shape)
    else:
        load_testsets(data_path, patch_shape=args.patch_shape)


if __name__ == "__main__":
    main()
