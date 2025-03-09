import os
import shutil
import random
import argparse
from glob import glob
import pathlib as Path
from natsort import natsorted

import numpy as np
import imageio.v3 as imageio

from micro_sam.util import get_sam_model


PADDING_DS = ["pannuke", "srsanet", "nuclick"]

DATASETS = [
    "consep",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "glas",
    "lizard",
    "lynsec_he",
    "lynsec_ihc",
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
    file_list = natsorted(glob(os.path.join(path, "labels", "*")))
    for label_path in file_list:
        label = imageio.imread(label_path)
        unique_elements = np.unique(label)
        if len(unique_elements) == 1:
            print(f"Image {os.path.basename(label_path)} does not contain labels and will be removed.")
            empty_count += 1
            os.remove(label_path)
            os.remove(os.path.join(path, "images", f"{os.path.basename(label_path)}"))
            assert len(os.listdir(os.path.join(path, "labels"))) == len(os.listdir(os.path.join(path, "images")))

    print(f"{empty_count} labels were empty")
    print(f"There are {len(os.listdir(os.path.join(path, 'labels')))} images left")


def create_val_split(
    path,
    val_percentage=0.05,
    test_percentage=0.95,
    custom_name="eval_split",
    random_seed=42,
    dataset=None,
):
    labels_src_path = os.path.join(path, "loaded_labels")
    images_src_path = os.path.join(path, "loaded_images")
    label_list = natsorted(glob(os.path.join(labels_src_path, "*")))
    if len(label_list) == 0:
        print(f"no labels found for {dataset}")
        return

    image_list = natsorted(glob(os.path.join(images_src_path, "*")))
    label_ext = os.path.splitext(os.path.basename(label_list[0]))[1]
    assert len(label_list) == len(image_list), "Mismatch in labels and images"
    splits = ["val", "test", "train"]
    dst_paths = {f"{split}_labels": Path(path) / custom_name / f"{split}_labels" for split in splits}
    dst_paths.update({f"{split}_images": Path(path) / custom_name / f"{split}_images" for split in splits})
    for dst in dst_paths.values():
        dst.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if len(os.listdir(os.path.join(path, custom_name, f"{split}_images"))) > 0:
            print(f"Split for {dataset} already exists")
            return

    print(f"No pre-existing validation or test set for {dataset} was found. A validation set will be created.")
    val_count = max(round(len(image_list) * val_percentage), 1)
    test_count = len(image_list) - val_count
    print(
        f"The validation set of {dataset} will consist of {val_count} images. \n"
        f"The test set of {dataset} will consist of {test_count} images."
    )

    random.seed(random_seed)
    val_indices = random.sample(range(0, (len(image_list))), val_count)
    val_images = [image_list[x] for x in val_indices]
    for val_image in val_images:
        label_name = os.path.basename(val_image).replace(os.path.splitext(os.path.basename(val_image))[1], label_ext)
        label_path = os.path.join(labels_src_path, label_name)
        shutil.copy(val_image, dst_paths["val_images"])
        shutil.copy(label_path, dst_paths["val_labels"])
        image_list.remove(val_image)
        label_list.remove(os.path.join(labels_src_path, label_name))
    assert len(os.listdir(dst_paths["val_labels"])) == len(os.listdir(dst_paths["val_images"])), \
        "label / image count mismatch in val set"

    test_indices = random.sample(range(0, (len(image_list))), test_count)
    if test_count > 0:
        test_images = [image_list[x] for x in test_indices]
        test_images.sort(reverse=True)
        for test_image in test_images:
            label_name = os.path.basename(test_image).replace(os.path.splitext(
                os.path.basename(test_image))[1], label_ext)
            label_path = os.path.join(labels_src_path, label_name)
            image_list.remove(test_image)
            label_list.remove(os.path.join(labels_src_path, label_name))
            shutil.copy(test_image, dst_paths["test_images"])
            shutil.copy(label_path, dst_paths["test_labels"])

    assert len(os.listdir(dst_paths["test_labels"])) == len(os.listdir(dst_paths["test_images"])), \
        "label / image count mismatch in test set"

    # residual images are per default in the train set
    for train_image in image_list:
        label_path = os.path.join(labels_src_path, os.path.splitext(os.path.basename(val_image))[0])
        shutil.copy(train_image, dst_paths["train_images"])
        shutil.copy(label_path, dst_paths["train_labels"])

    assert len(os.listdir(dst_paths["train_labels"])) == len(os.listdir(dst_paths["train_images"])), \
        "label / image count mismatch in val set"
    print(
        f"Train set: {len(os.listdir(dst_paths['train_images']))} images; "
        f" val set: {len(os.listdir(dst_paths['val_images']))} images; "
        f"test set: {len(os.listdir(dst_paths['test_images']))}"
    )


def get_model(model_type, ckpt):
    predictor = get_sam_model(model_type=model_type, checkpoint_path=ckpt)
    return predictor


def get_pred_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*")))
    return pred_paths


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=none_or_str, default=None)  # expects best.pt
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)  # empty directory for saving the output
    parser.add_argument(
        "-i", "--input_path", type=str, required=True, default=None,
        help="requires path to a directory containing 'test_images', 'test_labels', 'val_images' \
            and 'val_labels' directories that contain the data",
    )
    parser.add_argument("--organ", type=str, required=False, default=None)  # to access organ class or all dataset.
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
    parser.add_argument("--use_masks", action="store_true", help="To use logits masks for iterative prompting.")
    args = parser.parse_args()
    return args


def dataloading_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-d", "--datasets", type=str, default=None)
    parser.add_argument("--patch_shape", type=tuple, default=(512, 512))

    args = parser.parse_args()
    return args


def none_or_str(value):
    if value == "None":
        return None

    return value


def get_val_paths(input_path):
    val_image_paths = natsorted(glob(os.path.join(input_path, "val_images/*")))
    val_label_paths = natsorted(glob(os.path.join(input_path, "val_labels/*")))
    return val_image_paths, val_label_paths


def get_test_paths(input_path):
    test_image_paths = natsorted(glob(os.path.join(input_path, "test_images/*")))
    test_label_paths = natsorted(glob(os.path.join(input_path, "test_labels/*")))
    return test_image_paths, test_label_paths
