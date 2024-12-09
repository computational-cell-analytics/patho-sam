import os
import argparse
from glob import glob
from micro_sam.util import get_sam_model
from natsort import natsorted

ROOT = "/scratch/projects/nim00007/sam/data/"

EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models"

VANILLA_MODELS = {
    "vit_t": "/scratch-grete/projects/nim00007/sam/models/vanilla/vit_t_mobile_sam.pth",
    "vit_b": "/scratch-grete/projects/nim00007/sam/models/vanilla/sam_vit_b_01ec64.pth",
    "vit_l": "/scratch-grete/projects/nim00007/sam/models/vanilla/sam_vit_l_0b3195.pth",
    "vit_h": "/scratch-grete/projects/nim00007/sam/models/vanilla/sam_vit_h_4b8939.pth"
}


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
    parser.add_argument("-c", "--checkpoint", type=none_or_str, default=None) #expects best.pt
    parser.add_argument("-e", "--experiment_folder", type=str, required=True) #empty directory for saving the output
    parser.add_argument("-d", "--dataset", type=str, required=True, default=None) #defines dataset, mandatory
    parser.add_argument("-i", "--input_path", type=str, required=True, default=None)
    parser.add_argument("--organ", type=str, required=False, default=None) #optionally defines organ class to access. If empty, whole dataset is used
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box") #otherwise, point
    parser.add_argument(
        "--use_masks", action="store_true", help="To use logits masks for iterative prompting." 
    )
    args = parser.parse_args()
    return args

def dataloading_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--datasets", type=str, default=None,
    )
    parser.add_argument(
        "-p", "--path", type=str, default=None
    )
    args = parser.parse_args()
    return args


def none_or_str(value):
    if value == 'None':
        return None
    return value


def get_val_paths(input_path, dataset):
    path = os.path.join(input_path, dataset, 'loaded_dataset/complete_dataset/standard_split')
    val_image_paths = natsorted(glob(os.path.join(path, 'val_images/*')))
    val_label_paths = natsorted(glob(os.path.join(path, 'val_labels/*')))
    print(len(val_image_paths), len(val_label_paths))

    return val_image_paths, val_label_paths


def get_test_paths(input_path, dataset):
    path = os.path.join(input_path, dataset, 'loaded_dataset/complete_dataset/standard_split')
    test_image_paths = natsorted(glob(os.path.join(path, 'test_images/*')))
    test_label_paths = natsorted(glob(os.path.join(path, 'test_labels/*')))
    print(len(test_image_paths), len(test_label_paths))
    return test_image_paths, test_label_paths
