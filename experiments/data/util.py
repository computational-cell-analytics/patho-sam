import argparse
import os
from glob import glob

from micro_sam.util import get_sam_model
from natsort import natsorted

ROOT = "/scratch/projects/nim00007/sam/data/"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models"

VANILLA_MODELS = {
    "vit_t": "/scratch-grete/projects/nim00007/sam/models/vanilla/vit_t_mobile_sam.pth",
    "vit_b": "/scratch-grete/projects/nim00007/sam/models/vanilla/sam_vit_b_01ec64.pth",
    "vit_l": "/scratch-grete/projects/nim00007/sam/models/vanilla/sam_vit_l_0b3195.pth",
    "vit_h": "/scratch-grete/projects/nim00007/sam/models/vanilla/sam_vit_h_4b8939.pth",
}

PADDING_DS = [
    "pannuke",
    "srsanet",
    "nuclick",
]

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


def get_dataset_paths(dataset_name, split_choice):
    file_search_specs = "*"
    is_explicit_split = True

    # if the datasets have different modalities/species, let's make use of it
    split_names = dataset_name.split("/")
    if len(split_names) > 1:
        assert len(split_names) <= 2
        dataset_name = [split_names[0], "slices", split_names[1]]
    else:
        dataset_name = [*split_names, "slices"]

    # if there is an explicit val/test split made, let's look at them
    if is_explicit_split:
        dataset_name.append(split_choice)

    raw_dir = os.path.join(ROOT, *dataset_name, "raw", file_search_specs)
    labels_dir = os.path.join(ROOT, *dataset_name, "labels", file_search_specs)

    return raw_dir, labels_dir


def get_model(model_type, ckpt):
    if ckpt is None:
        ckpt = VANILLA_MODELS[model_type]
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
    parser.add_argument("-ps", "--patch_shape", type=tuple, default=(512, 512))

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
