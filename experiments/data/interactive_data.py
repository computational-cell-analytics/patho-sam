import os
import argparse

import imageio.v3 as imageio

from dataloaders import get_dataloaders
from util import DATASETS
from preprocess_datasets import create_val_split


def load_testsets(path, dsets=DATASETS, patch_shape=(512, 512)) -> None:
    for dataset in dsets:
        if os.path.exists(os.path.join(path, dataset, "loaded_testset", "images")):
            if len(os.listdir(os.path.join(path, dataset, "loaded_testset", "images"))) > 1:
                print(f"Dataset {dataset} is loaded already.")
                continue
        counter = 1
        print(f"Loading {dataset} dataset...")
        dpath = os.path.join(path, dataset)
        os.makedirs(dpath, exist_ok=True)
        loader = get_dataloaders(patch_shape=patch_shape, data_path=dpath, dataset=dataset)
        image_output_path = os.path.join(path, dataset, "loaded_testset", "images")
        label_output_path = os.path.join(path, dataset, "loaded_testset", "labels")
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)
        for image, label in loader:
            image_array = image.numpy()
            label_array = label.numpy()
            squeezed_image = image_array.squeeze()
            label_data = label_array.squeeze()
            tp_img = squeezed_image.transpose(1, 2, 0)
            if tp_img.shape[-1] == 4:  # deletes alpha channel if one exists
                tp_img = tp_img[..., :-1]
            imageio.imwrite(os.path.join(image_output_path, f"{counter:04}.tiff"), tp_img)
            imageio.imwrite(os.path.join(label_output_path, f"{counter:04}.tiff"), label_data)
            counter += 1
        create_val_split(os.path.join(dpath, "loaded_testset"), custom_name="eval_split", dataset=dataset)
        print(f"{dataset} testset has successfully been loaded.")


def dataloading_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-d", "--datasets", type=str, default=None)
    parser.add_argument("-ps", "--patch_shape", type=tuple, default=(512, 512))

    args = parser.parse_args()
    return args


def main():
    args = dataloading_args()
    if args.path is not None:
        data_path = args.path
    else:
        data_path = "/mnt/lustre-grete/usr/u12649/data/final_test/"

    if args.datasets is not None:
        load_testsets(data_path, [args.datasets], args.patch_shape)
    else:
        load_testsets(data_path, patch_shape=args.patch_shape)


if __name__ == "__main__":
    main()
