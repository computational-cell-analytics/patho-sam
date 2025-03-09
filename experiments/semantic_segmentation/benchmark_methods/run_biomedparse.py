import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from tukra.io import read_image
from tukra.inference import get_biomedparse

from patho_sam.evaluation.evaluation import semantic_segmentation_quality, extract_class_weights_for_pannuke


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/semantic/external"

MAPPING = {
    "neoplastic cells": 1,
    "inflammatory cells": 2,
    "connective tissue cells": 3,
    "dead cells": 4,  # NOTE: dead cells are not a semantic class involved in biomedparse.
    "epithelial cells": 5,
}


def evaluate_biomedparse_for_histopathology(dataset):

    # Other stuff for biomedparse
    modality = "Pathology"  # choice of modality to determine the semantic targets.
    model = get_biomedparse.get_biomedparse_model()  # get the biomedparse model.

    # Get per class weights.
    fpath = os.path.join(*ROOT.rsplit("/")[:-2], "data", "pannuke", "pannuke_fold_3.h5")
    fpath = "/" + fpath
    per_class_weights = extract_class_weights_for_pannuke(fpath=fpath)

    # Get the inputs and corresponding labels.
    image_paths = natsorted(glob(os.path.join(ROOT, dataset, "test_images", "*.tiff")))
    gt_paths = natsorted(glob(os.path.join(ROOT, dataset, "test_labels", "*.tiff")))

    assert image_paths and len(image_paths) == len(gt_paths)

    # Get results directory
    result_dir = os.path.join(ROOT, "biomedparse", dataset)
    os.makedirs(result_dir, exist_ok=True)

    sq_per_image = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        # Get the input image and corresponding semantic labels.
        image = read_image(image_path).astype("float32")
        gt = read_image(gt_path)

        # If the inputs do not have any semantic labels, we do not evaluate them!
        if len(np.unique(gt)) == 1:
            continue

        # Run inference per image.
        prediction = get_biomedparse.run_biomedparse_automatic_inference(
            input_path=image, modality_type=modality, model=model, verbose=False,
        )

        semantic_seg = np.zeros_like(gt, dtype="uint8")
        if prediction is not None:
            prompts = list(prediction.keys())  # Extracting detected classes.
            segmentations = list(prediction.values())  # Extracting the segmentations.

            # Map all predicted labels.
            for prompt, per_seg in zip(prompts, segmentations):
                semantic_seg[per_seg > 0] = MAPPING[prompt]

        # Evaluate scores.
        sq_score = semantic_segmentation_quality(
            ground_truth=gt, segmentation=semantic_seg, class_ids=list(MAPPING.values()),
        )
        sq_per_image.append(sq_score)

        # Store the semantic segmentation results to avoid running them all the time.
        imageio.imwrite(os.path.join(result_dir, os.path.basename(image_path)), semantic_seg, compression="zlib")

    def _get_average_results(sq_per_image, fname):
        msq_neoplastic_cells = np.nanmean([sq[0] for sq in sq_per_image])
        msq_inflammatory = np.nanmean([sq[1] for sq in sq_per_image])
        msq_connective = np.nanmean([sq[2] for sq in sq_per_image])
        msq_dead = np.nanmean([sq[3] for sq in sq_per_image])  # This class is absent. So this would obv. be zero.
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

    # Get average results for biomedparse.
    _get_average_results(sq_per_image, "biomedparse_semantic.csv")


def main():
    # Run automatic (semantic) segmentation inference using biomedparse
    evaluate_biomedparse_for_histopathology("pannuke")
    evaluate_biomedparse_for_histopathology("puma")


if __name__ == "__main__":
    main()
