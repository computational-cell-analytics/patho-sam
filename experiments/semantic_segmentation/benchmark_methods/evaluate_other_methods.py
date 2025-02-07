import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np
import pandas as pd

from tukra.io import read_image

from patho_sam.evaluation.evaluation import semantic_segmentation_quality, extract_class_weights_for_pannuke


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/semantic/external"

CLASS_IDS = [1, 2, 3, 4, 5]


def evaluate_benchmark_methods(per_class_weights):
    # Get the original images first.
    image_paths = natsorted(glob(os.path.join(ROOT, "semantic_split", "test_images", "*.tiff")))
    gt_paths = natsorted(glob(os.path.join(ROOT, "semantic_split", "test_labels", "*.tiff")))

    assert image_paths and len(image_paths) == len(gt_paths)

    cellvit_256_20_scores, cellvit_256_40_scores, cellvit_sam_20_scores, cellvit_sam_40_scores = [], [], [], []
    hovernet_scores = []
    hovernext_1_scores, hovernext_2_scores = [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        # Load the input image and corresponding labels.
        gt = read_image(gt_path)

        # If the inputs do not have any semantic labels, we do not evaluate them!
        if len(np.unique(gt)) == 1:
            continue

        # Get the filename
        fname = os.path.basename(image_path)

        # Get predictions and scores per experiment.
        # 1. cellvit results (for all models).
        cellvit_256_20 = read_image(os.path.join(ROOT, "cellvit", "256-x20", fname))
        cellvit_256_40 = read_image(os.path.join(ROOT, "cellvit", "256-x40", fname))
        cellvit_sam_20 = read_image(os.path.join(ROOT, "cellvit", "SAM-H-x20", fname))
        cellvit_sam_40 = read_image(os.path.join(ROOT, "cellvit", "SAM-H-x40", fname))

        cellvit_256_20_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=cellvit_256_20, class_ids=CLASS_IDS)
        )
        cellvit_256_40_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=cellvit_256_40, class_ids=CLASS_IDS)
        )
        cellvit_sam_20_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=cellvit_sam_20, class_ids=CLASS_IDS)
        )
        cellvit_sam_40_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=cellvit_sam_40, class_ids=CLASS_IDS)
        )

        # 2. hovernet results.
        hovernet = read_image(os.path.join(ROOT, "hovernet", "pannuke", fname))

        hovernet_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=hovernet, class_ids=CLASS_IDS)
        )

        # 3. hovernext results.
        hovernext_1 = read_image(os.path.join(ROOT, "hovernext", "pannuke_convnextv2_tiny_1", fname))
        hovernext_2 = read_image(os.path.join(ROOT, "hovernext", "pannuke_convnextv2_tiny_2", fname))

        hovernext_1_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=hovernext_1, class_ids=CLASS_IDS)
        )
        hovernext_2_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=hovernext_2, class_ids=CLASS_IDS)
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
    _get_average_results(cellvit_256_20_scores, "cellvit_256_20_semantic.csv")
    _get_average_results(cellvit_256_40_scores, "cellvit_256_40_semantic.csv")
    _get_average_results(cellvit_sam_20_scores, "cellvit_sam_20_semantic.csv")
    _get_average_results(cellvit_sam_40_scores, "cellvit_sam_40_semantic.csv")
    _get_average_results(hovernet_scores, "hovernet_semantic.csv")
    _get_average_results(hovernext_1_scores, "hovernext_1_semantic.csv")
    _get_average_results(hovernext_2_scores, "hovernext_2_semantic.csv")


def main():
    # Get per class weights.
    fpath = os.path.join(*ROOT.rsplit("/")[:-2], "data", "pannuke", "pannuke_fold_3.h5")
    fpath = "/" + fpath
    per_class_weights = extract_class_weights_for_pannuke(fpath=fpath)

    # Run evaluation for external benchmark methods.
    evaluate_benchmark_methods(per_class_weights)


if __name__ == "__main__":
    main()
