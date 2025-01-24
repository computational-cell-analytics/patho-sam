import os
import shutil
from glob import glob
import argparse

from get_paths import get_dataset_paths
from util import DATASETS


def load_datasets(path, datasets=DATASETS):
    for dataset in datasets:
        dataset_path = os.path.join(path, dataset)
        image_outpath = os.path.join(path, dataset, "loaded_images")
        label_outpath = os.path.join(path, dataset, "loaded_images")
        os.makedirs(image_outpath, exist_ok=True)
        os.makedirs(label_outpath, exist_ok=True)
        image_paths, label_paths = get_dataset_paths(dataset_path, dataset)
        count = 1
        for image_path, label_path in zip(image_paths, label_paths):
            img_ext = os.path.splitext(image_path)[1]
            label_ext = os.path.splitext(label_path)[1]
            image_dest = os.path.join(image_outpath, f"{count:04}{img_ext}")
            label_dest = os.path.join(label_outpath, f"{count:04}{label_ext}")

            shutil.move(image_path, image_dest)
            shutil.move(label_path, label_dest)


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
        data_path = "/mnt/lustre-grete/usr/u12649/data/original_data/"

    if args.datasets is not None:
        load_datasets(data_path, [args.datasets])
   
    else:
        load_datasets(data_path)


if __name__ == "__main__":
    main()
