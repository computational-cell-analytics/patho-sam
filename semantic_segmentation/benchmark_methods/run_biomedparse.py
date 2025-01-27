import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tukra.io import read_image
from tukra.inference import get_biomedparse

from patho_sam.evaluation import semantic_segmentation_quality


MAPPING = {
    "neoplastic cells": 1,
    "inflammatory cells": 2,
    "connective tissue cells": 3,
    # NOTE: dead cells are not a semantic class involved in biomedparse.
    "epithelial cells": 5,
}


def evaluate_biomedparse_for_pannuke(input_path, view):
    # Other stuff for biomedparse
    modality = "Pathology"  # choice of modality to determine the semantic targets.
    model = get_biomedparse.get_biomedparse_model()  # get the biomedparse model.

    # Get the inputs and corresponding labels.
    image_paths = natsorted(glob(os.path.join(input_path, "pannuke", "fold3_eval", "test_images", "*")))
    gt_paths = natsorted(glob(os.path.join(input_path, "pannuke", "fold3_eval", "test_labels", "*")))

    sq_per_image = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        # Get the input image and corresponding semantic labels.
        image = read_image(image_path)
        gt = read_image(gt_path)

        # Get the valid region as the remaining is padded.
        one_chan = image[:, :, 0]  # Take one channel to extract valid channels.
        idxx = np.argwhere(one_chan > 0)
        x_min, y_min = idxx.min(axis=0)
        x_max, y_max = idxx.max(axis=0)

        # Crop out valid region in image and corresponding labels and segmentation.
        image = image[x_min:x_max+1, y_min:y_max+1]
        gt = gt[x_min:x_max+1, y_min:y_max+1]

        # Run inference per image.
        prediction = get_biomedparse.run_biomedparse_automatic_inference(
            input_path=image, modality_type=modality, model=model, verbose=False,
        )

        prompts = list(prediction.keys())  # Extracting detected classes.
        segmentations = list(prediction.values())  # Extracting the segmentations.

        # Map all predicted labels.
        semantic_seg = np.zeros_like(gt, dtype="uint8")
        for prompt, per_seg in zip(prompts, segmentations):
            semantic_seg[per_seg > 0] = MAPPING[prompt]

        # Evaluate scores.
        sq_score = semantic_segmentation_quality(
            ground_truth=gt, segmentation=semantic_seg, class_ids=[1, 2, 3, 4, 5]
        )
        sq_per_image.append(sq_score)

        if view:
            fig, axes = plt.subplots(1, 3, figsize=(20, 10))
            axes[0].imshow(image.astype(int))
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(gt)
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')

            axes[2].imshow(semantic_seg)
            axes[2].set_title("Predictions")
            axes[2].axis('off')

            plt.savefig("./test.png", bbox_inches="tight")
            plt.close()

            breakpoint()

    msq_neoplastic_cells = np.nanmean([sq[0] for sq in sq_per_image])
    msq_inflammatory = np.nanmean([sq[1] for sq in sq_per_image])
    msq_connective = np.nanmean([sq[2] for sq in sq_per_image])
    msq_epithelial = np.nanmean([sq[3] for sq in sq_per_image])

    results = {
        "neoplastic_cells": msq_neoplastic_cells,
        "inflammatory_cells": msq_inflammatory,
        "connective_cells": msq_connective,
        "epithelial_cells": msq_epithelial,
        "mean": np.mean([msq_neoplastic_cells, msq_inflammatory, msq_connective, msq_epithelial]),
    }
    results = pd.DataFrame.from_dict([results])
    print(results)


def main(args):
    # Run automatic (semantic) segmentation inference using biomedparse
    evaluate_biomedparse_for_pannuke(args.input_path, args.view)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/test/data", type=str)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    main(args)
