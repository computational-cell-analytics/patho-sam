import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import h5py
import numpy as np
import pandas as pd
import imageio.v3 as imageio

from patho_sam.evaluation import semantic_segmentation_quality


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/semantic/external"

CLASS_IDS = [1, 2, 3, 4, 5]


def _get_class_weights_for_pannuke():
    stack_path = os.path.join(*ROOT.rsplit("/")[:-2], "data", "pannuke", "pannuke_fold_3.h5")
    stack_path = "/" + stack_path

    # Load the entire instance and semantic stack.
    with h5py.File(stack_path, "r") as f:
        instances = f['labels/instances'][:]
        semantic = f['labels/semantic'][:]

    # We need the following:
    # - Count the total number of instances.
    total_instance_counts = [
        len(np.unique(ilabel)[1:]) for ilabel in instances if len(np.unique(ilabel)) > 1
    ]  # Counting all valid foreground instances only.
    total_instance_counts = sum(total_instance_counts)

    # - Count per-semantic-class instances.
    total_per_class_instance_counts = [
        [len(np.unique(np.where(slabel == cid, ilabel, 0))[1:]) for cid in CLASS_IDS]
        for ilabel, slabel in zip(instances, semantic) if len(np.unique(ilabel)) > 1
    ]
    assert total_instance_counts == sum([sum(t) for t in total_per_class_instance_counts])

    # Calculate per class mean values.
    total_per_class_instance_counts = [sum(x) for x in zip(*total_per_class_instance_counts)]
    assert total_instance_counts == sum(total_per_class_instance_counts)

    # Finally, let's get the weight per class.
    per_class_weights = [t / total_instance_counts for t in total_per_class_instance_counts]

    return per_class_weights


def evaluate_benchmark_methods(per_class_weights):
    # Get the original images first.
    image_paths = natsorted(glob(os.path.join(ROOT, "semantic_split", "test_images", "*.tiff")))
    gt_paths = natsorted(glob(os.path.join(ROOT, "semantic_split", "test_labels", "*.tiff")))

    assert image_paths and len(image_paths) == len(gt_paths)

    cellvit_scores, hovernet_scores, hovernext_scores = [], [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        # Load the input image and corresponding labels.
        # image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        # If the inputs do not have any semantic labels, we do not evaluate them!
        if len(np.unique(gt)) == 1:
            continue

        # Get the filename
        fname = os.path.basename(image_path)

        # Get predictions per experiment.
        cellvit = imageio.imread(os.path.join(ROOT, "cellvit", "SAM-H-x40", fname))
        hovernet = imageio.imread(os.path.join(ROOT, "hovernet", "pannuke", fname))
        hovernext = imageio.imread(os.path.join(ROOT, "hovernext", "pannuke_convnextv2_tiny_1", fname))

        # Get scores per experiment.
        cellvit_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=cellvit, class_ids=CLASS_IDS)
        )
        hovernet_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=hovernet, class_ids=CLASS_IDS)
        )
        hovernext_scores.append(
            semantic_segmentation_quality(ground_truth=gt, segmentation=hovernext, class_ids=CLASS_IDS)
        )

    def _get_average_results(sq_per_image):
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
            "weighted_mean": np.mean(weighted_mean_msq),
            "absolute_mean": np.mean(all_msq)
        }
        results = pd.DataFrame.from_dict([results])
        print(results)

    # Get average results per method.
    _get_average_results(cellvit_scores)
    _get_average_results(hovernet_scores)
    _get_average_results(hovernext_scores)


def main():
    per_class_weights = _get_class_weights_for_pannuke()
    evaluate_benchmark_methods(per_class_weights)


if __name__ == "__main__":
    main()
