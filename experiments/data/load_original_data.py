import os
import shutil
import argparse
from tqdm import tqdm

import imageio.v3 as imageio

from util import DATASETS
from get_paths import get_dataset_paths


def load_datasets(path, datasets=DATASETS):
    for dataset in datasets:
        dataset_path = os.path.join(path, dataset)
        image_outpath = os.path.join(dataset_path, "loaded_images")
        label_outpath = os.path.join(dataset_path, "loaded_labels")
        image_outpath = os.path.join(dataset_path, "loaded_images")
        label_outpath = os.path.join(dataset_path, "loaded_labels")
        os.makedirs(image_outpath, exist_ok=True)
        os.makedirs(label_outpath, exist_ok=True)
        if len(os.listdir(image_outpath)) > 1:
            continue

        print(f"Loading {dataset}...")
        if len(os.listdir(image_outpath)) > 1:
            continue

        print(f"Loading {dataset}...")
        image_paths, label_paths = get_dataset_paths(dataset_path, dataset)
        assert len(image_paths) == len(label_paths)

        count = 1
        for image_path, label_path in tqdm(zip(image_paths, label_paths), desc="Moving files to new directory..."):
            img_ext = os.path.splitext(image_path)[1]
            label_ext = os.path.splitext(label_path)[1]
            image_dest = os.path.join(image_outpath, f"{count:04}{img_ext}")
            label_dest = os.path.join(label_outpath, f"{count:04}{label_ext}")

            img = imageio.imread(image_path)
            if img.shape[2] == 4:  # checks for and removes alpha channel
                img = img[:, :, :-1]
                imageio.imwrite(image_path, img)
                print("Alpha channel successfully removed.")

            shutil.move(image_path, image_dest)
            shutil.move(label_path, label_dest)

            count += 1


def dataloading_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-d", "--datasets", type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = dataloading_args()
    if args.path is not None:
        data_path = args.path
    else:
        data_path = "/mnt/lustre-grete/usr/u12649/data/original_data"

    if args.datasets is not None:
        load_datasets(data_path, [args.datasets])
    else:
        load_datasets(data_path)


if __name__ == "__main__":
    main()
