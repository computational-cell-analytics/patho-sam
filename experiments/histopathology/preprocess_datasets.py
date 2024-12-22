import os
import random
import shutil
from glob import glob
from pathlib import Path

import numpy as np
from natsort import natsorted
from skimage import io

""" Script to create:
- a reproducable validation, training and test set from the root directory 
- evaluation directories for the segmentation mode: automatic instance segmentation, automatic mask generation, iterative prompting with and without boxes for each given model
- if prompt is set to True, labels without instances (just background) are removed in order for the iterative prompting evaluation to work
"""  # noqa


def remove_empty_labels(path):
    empty_count = 0
    file_list = natsorted(glob(os.path.join(path, "*.tiff")))
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


def create_val_split(
    directory,
    val_percentage=0.05,
    test_percentage=0.95,
    custom_name="standard_split",
    organ_type=None,
    split=None,
    random_seed=42,
):
    if split is None:
        path = os.path.join(directory, "loaded_dataset/complete_dataset")
    else:
        path = os.path.join(directory, split)
    labels_src_path = Path(path) / "labels"
    images_src_path = Path(path) / "images"
    label_list = natsorted(glob(str(labels_src_path / "*.tiff")))
    image_list = natsorted(glob(str(images_src_path / "*.tiff")))
    assert len(label_list) == len(image_list), "Mismatch in labels and images"
    splits = ["val", "test", "train"]
    dst_paths = {f"{split}_labels": Path(path) / custom_name / f"{split}_labels" for split in splits}
    dst_paths.update({f"{split}_images": Path(path) / custom_name / f"{split}_images" for split in splits})
    for dst in dst_paths.values():
        dst.mkdir(parents=True, exist_ok=True)
    for split in splits:
        # assert not list(dst_paths[f"{split}_labels"].iterdir()), f"{split.capitalize()} labels split already exists"
        # assert not list(dst_paths[f"{split}_images"].iterdir()), f"{split.capitalize()} images split already exists"
        if (
            len(
                os.listdir(
                    os.path.join(directory, "loaded_dataset", "complete_dataset", custom_name, f"{split}_images")
                )
            )
            > 0
        ):
            print("Split already exists")
            return

    print("No pre-existing validation or test set was found. A validation set will be created.")
    val_count = max(round(len(image_list) * val_percentage), 1)
    test_count = len(image_list) - val_count
    print(
        f"The validation set will consist of {val_count} images. \n"
        f"The test set will consist of {test_count} images."
    )

    random.seed(random_seed)
    val_indices = random.sample(range(0, (len(image_list))), val_count)
    val_images = [image_list[x] for x in val_indices]
    for val_image in val_images:
        label_path = os.path.join(labels_src_path, os.path.basename(val_image))
        shutil.copy(val_image, dst_paths["val_images"])
        shutil.copy(label_path, dst_paths["val_labels"])
        image_list.remove(val_image)
        label_list.remove((os.path.join(labels_src_path, (os.path.basename(val_image)))))
    assert len(os.listdir(dst_paths["val_labels"])) == len(
        os.listdir(dst_paths["val_images"])
    ), "label / image count mismatch in val set"

    test_indices = random.sample(range(0, (len(image_list))), test_count)
    if test_count > 0:
        test_images = [image_list[x] for x in test_indices]
        test_images.sort(reverse=True)
        for test_image in test_images:
            label_path = os.path.join(labels_src_path, os.path.basename(test_image))
            image_list.remove(test_image)
            label_list.remove((os.path.join(labels_src_path, (os.path.basename(test_image)))))
            shutil.copy(test_image, dst_paths["test_images"])
            shutil.copy(label_path, dst_paths["test_labels"])

    assert len(os.listdir(dst_paths["test_labels"])) == len(
        os.listdir(dst_paths["test_images"])
    ), "label / image count mismatch in val set"
    # residual images are per default in the train set
    for train_image in image_list:
        label_path = os.path.join(labels_src_path, os.path.basename(train_image))
        shutil.copy(train_image, dst_paths["train_images"])
        shutil.copy(label_path, dst_paths["train_labels"])
    assert len(os.listdir(dst_paths["train_labels"])) == len(
        os.listdir(dst_paths["train_images"])
    ), "label / image count mismatch in val set"
    print(
        f"Train set: {len(os.listdir(dst_paths['train_images']))} images; "
        f" val set: {len(os.listdir(dst_paths['val_images']))} images; "
        f"test set: {len(os.listdir(dst_paths['test_images']))}"
    )


def create_eval_directories(eval_path, dataset, models):
    datasets = [
        "cpm15",
        "cpm17",
        "cryonuseg",
        "janowczyk",
        "lizard",
        "lynsec",
        "monusac",
        "monuseg",
        "nuinsseg",
        "pannuke",
        "puma",
        "tnbc",
    ]
    for dataset in datasets:
        for mode in ["instance", "boxes", "points", "amg"]:
            os.makedirs(os.path.join(eval_path, dataset, mode), exist_ok=True)


create_eval_directories("/mnt/lustre-grete/usr/u12649/scratch/models/generalist_sam/inference")


def preprocess_datasets(eval_path, data_path, model_names=None, prompt=False):
    datasets = [
        "cpm15",
        "cpm17",
        "cryonuseg",
        "janowczyk",
        "lizard",
        "lynsec",
        "monusac",
        "monuseg",
        "nuinsseg",
        "pannuke",
        "puma",
        "tnbc",
    ]
    for dataset in datasets:
        print("Checking labels of dataset: ", dataset)
        remove_empty_labels(os.path.join(data_path, dataset, "loaded_dataset", "complete_dataset"))
        create_val_split(
            os.path.join(data_path, dataset),
            val_percentage=0.05,
            test_percentage=0.95,
            custom_name="eval_split",
            random_seed=42,
        )


# preprocess_datasets(' ', '/mnt/lustre-grete/usr/u12649/scratch/data/test')
