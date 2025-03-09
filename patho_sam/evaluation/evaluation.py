import os
from typing import List, Union

import h5py
import numpy as np

from elf.evaluation import dice_score

CLASS_IDS = [1, 2, 3, 4, 5]


def extract_class_weights_for_pannuke(fpath: Union[os.PathLike, str], class_ids: List = CLASS_IDS) -> List[float]:
    """Extract class weights per semantic class.

    Args:
        fpath: The filepath where the input stack for fold 3 stored for PanNuke dataset.
            Use `torch_em.data.datasets.histopathology.get_pannuke_paths` to get filepath for the stack.
        class_ids: The choice of all available class ids.

    Returns:
        List of class weights.
    """
    # Load the entire instance and semantic stack.
    with h5py.File(fpath, "r") as f:
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
        [len(np.unique(np.where(slabel == cid, ilabel, 0))[1:]) for cid in class_ids]
        for ilabel, slabel in zip(instances, semantic) if len(np.unique(ilabel)) > 1
    ]
    assert total_instance_counts == sum([sum(t) for t in total_per_class_instance_counts])

    # Calculate per class mean values.
    total_per_class_instance_counts = [sum(x) for x in zip(*total_per_class_instance_counts)]
    assert total_instance_counts == sum(total_per_class_instance_counts)

    # Finally, let's get the weight per class.
    per_class_weights = [t / total_instance_counts for t in total_per_class_instance_counts]

    return per_class_weights


def semantic_segmentation_quality(
    ground_truth: np.ndarray, segmentation: np.ndarray, class_ids: List[int]
) -> List[float]:
    """Evaluation metric for the semantic segmentation task.

    Args:
        ground_truth: The ground truth with expected semantic labels.
        segmentation: The predicted masks with expected semantic labels.
        class_ids: The per-class id available for all tasks, to calculate per class semantic quality score.

    Returns:
        List of semantic quality score per class.
    """
    # First, we iterate over all classes
    sq_per_class = []
    for id in class_ids:
        # Get the per semantic class values.
        this_gt = (ground_truth == id).astype("uint32")
        this_seg = (segmentation == id).astype("uint32")

        # Check if the ground truth is empty for this semantic class. We skip calculation for this.
        if len(np.unique(this_gt)) == 1 and len(np.unique(this_seg) == 1):
            this_sq = np.nan
        else:
            this_sq = dice_score(this_seg, this_gt)

        sq_per_class.append(this_sq)

    return sq_per_class
