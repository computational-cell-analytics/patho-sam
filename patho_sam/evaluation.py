from typing import List

import numpy as np

from elf.evaluation import dice_score


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
        if len(np.unique(this_gt)) == 1:
            this_sq = np.nan
        else:
            this_sq = dice_score(this_seg, this_gt)

        sq_per_class.append(this_sq)

    return sq_per_class
