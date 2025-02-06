import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np
import pandas as pd

import torch

from tukra.io import read_image

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_unetr

from patho_sam.evaluation.evaluation import semantic_segmentation_quality, extract_class_weights_for_pannuke


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/semantic/external"


def evaluate_pannuke_semantic_segmentation(args):
    # Stuff needed for inference
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    num_classes = 6  # available classes are [0, 1, 2, 3, 4, 5]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get per class weights.
    fpath = os.path.join(*ROOT.rsplit("/")[:-2], "data", "pannuke", "pannuke_fold_3.h5")
    fpath = "/" + fpath
    per_class_weights = extract_class_weights_for_pannuke(fpath=fpath)

    # Get the inputs and corresponding labels.
    image_paths = natsorted(glob(os.path.join(ROOT, "semantic_split", "test_images", "*.tiff")))
    gt_paths = natsorted(glob(os.path.join(ROOT, "semantic_split", "test_labels", "*.tiff")))

    assert len(image_paths) == len(gt_paths) and image_paths

    # Get the SAM model
    predictor = get_sam_model(model_type=model_type, device=device)

    # Get the UNETR model for semantic segmentation pipeline
    unetr = get_unetr(
        image_encoder=predictor.model.image_encoder, out_channels=num_classes, device=device,
    )

    # Load the model weights
    model_state = torch.load(checkpoint_path, map_location="cpu")["model_state"]
    unetr.load_state_dict(model_state)
    unetr.to(device)
    unetr.eval()

    sq_per_image = []
    with torch.no_grad():
        for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
            # Read input image and corresponding labels.
            image = read_image(image_path)
            gt = read_image(gt_path)

            # If the inputs do not have any semantic labels, we do not evaluate them!
            if len(np.unique(gt)) == 1:
                continue

            # Pad the input image to fit the trained image shape.
            # NOTE: We pad it to top-left.
            image = np.pad(array=image, pad_width=((0, 256), (0, 256), (0, 0)), mode='constant')

            # Run inference
            tensor_image = image.transpose(2, 0, 1)
            tensor_image = torch.from_numpy(tensor_image[None]).to(device, torch.float32)
            outputs = unetr(tensor_image)

            # Perform argmax to get per class outputs.
            masks = torch.argmax(outputs, dim=1)
            masks = masks.detach().cpu().numpy().squeeze()

            # Unpad the images back to match the original shape.
            masks = masks[:256, :256]

            # Calcuate the score.
            sq_per_image.append(
                semantic_segmentation_quality(gt, masks, class_ids=[1, 2, 3, 4, 5])
            )

    def _get_average_results(sq_per_image, fname):
        msq_neoplastic_cells = np.nanmean([sq[0] for sq in sq_per_image])
        msq_inflammatory = np.nanmean([sq[1] for sq in sq_per_image])
        msq_connective = np.nanmean([sq[2] for sq in sq_per_image])
        msq_dead = np.nanmean([sq[3] for sq in sq_per_image])
        msq_epithelial = np.nanmean([sq[4] for sq in sq_per_image])

        all_msq = [msq_neoplastic_cells, msq_inflammatory, msq_connective, msq_dead, msq_epithelial]
        weighted_mean_msq = [msq * weight for msq, weight in zip(all_msq, per_class_weights)]

        results = {
            "neoplastic_cells": msq_neoplastic_cells,
            "inflammatory_cells": msq_inflammatory,
            "connective_cells": msq_connective,
            "dead_cells": msq_dead,
            "epithelial_cells": msq_epithelial,
            "weighted_mean": np.sum(weighted_mean_msq),
            "absolute_mean": np.mean(all_msq)
        }
        results = pd.DataFrame.from_dict([results])
        results.to_csv(fname)
        print(results)

    # Get average results per method.
    fname = checkpoint_path.rsplit("/")[-2]  # Fetches the name of the style of training for semantic segmentation.
    _get_average_results(sq_per_image, f"pathosam_{fname}.csv")


def main(args):
    evaluate_pannuke_semantic_segmentation(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/test/data", type=str)
    parser.add_argument("-m", "--model_type", default="vit_b", type=str)
    parser.add_argument("-c", "--checkpoint_path", required=True)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    main(args)
