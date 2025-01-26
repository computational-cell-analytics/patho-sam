import os
import argparse

import imageio.v3 as imageio

from dataloaders import get_dataloaders
from util import DATASETS, create_val_split, remove_empty_labels


def load_testsets(path, dsets=DATASETS, patch_shape=(512, 512)) -> None:
    for dataset in dsets:
        if os.path.exists(os.path.join(path, dataset, "loaded_testset", "images")):
            if len(os.listdir(os.path.join(path, dataset, "loaded_testset", "images"))) > 1:
                print(f"Dataset {dataset} is loaded already.")
                continue

        print(f"Loading {dataset} dataset...")
        dpath = os.path.join(path, dataset)
        os.makedirs(dpath, exist_ok=True)
        loader = get_dataloaders(patch_shape=patch_shape, data_path=dpath, dataset=dataset)

        image_output_path = os.path.join(path, dataset, "loaded_testset", "images")
        label_output_path = os.path.join(path, dataset, "loaded_testset", "labels")

        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)

        for idx, (image, label) in enumerate(loader, start=1):
            image = image.squeeze().numpy()
            label = label.squeeze().numpy()
            image = image.transpose(1, 2, 0)
            if image.shape[-1] == 4:  # deletes alpha channel if one exists
                image = image[..., :-1]

            imageio.imwrite(os.path.join(image_output_path, f"{idx:04}.tiff"), image)
            imageio.imwrite(os.path.join(label_output_path, f"{idx:04}.tiff"), label)

        remove_empty_labels(dpath)
        create_val_split(os.path.join(dpath, "loaded_testset"), custom_name="eval_split", dataset=dataset)
        print(f"{dataset} testset has successfully been loaded.")


def dataloading_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-d", "--datasets", type=str, default=None)
    parser.add_argument("--patch_shape", type=tuple, default=(512, 512))

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
