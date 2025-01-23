import os
from glob import glob
from tqdm import tqdm
from typing import List
from natsort import natsorted

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch

from tukra.io import read_image

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_unetr


def get_fast_pq(true, pred, match_iou=0.5):
    """Inspired by:
    https://github.com/TissueImageAnalytics/PanNuke-metrics/blob/master/run.py

    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement
    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list) - 1,
                             len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id-1, pred_id-1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        # Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        # extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def panoptic_quality(ground_truth: np.ndarray, segmentation: np.ndarray, class_ids: List[int]):
    """Inspired by:
    https://github.com/TissueImageAnalytics/PanNuke-metrics/blob/master/run.py
    """
    # First, we iterate over all classes
    pq_per_class = []
    for i in class_ids:
        # Get the per semantic class values.
        this_gt = (ground_truth == i).astype("uint32")
        this_seg = (segmentation == i).astype("uint32")

        # Check if the ground truth is empty for this semantic class. We skip calculation for this.
        if len(np.unique(this_gt)) == 1:
            this_pq = np.nan
        else:
            # Computes PQ.
            [_, _, this_pq], _ = get_fast_pq(this_gt, this_seg)

        pq_per_class.append(this_pq)

    return pq_per_class


def evaluate_pannuke_semantic_segmentation(args):
    # Stuff needed for inference
    model_type = args.model_type
    num_classes = 6  # available classes are [0, 1, 2, 3, 4, 5]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the inputs and corresponding labels.
    image_paths = natsorted(glob(os.path.join(args.input_path, "pannuke", "fold3_eval", "test_images", "*")))
    gt_paths = natsorted(glob(os.path.join(args.input_path, "pannuke", "fold3_eval", "test_labels", "*")))

    assert len(image_paths) == len(gt_paths) and image_paths

    # Get the SAM model
    # NOTE: The users can pass `vit_b_histopathology` as it is supported in `micro-sam` (on `dev`).
    predictor = get_sam_model(model_type=model_type, device=device)

    # Get the UNETR model for semantic segmentation pipeline
    unetr = get_unetr(
        image_encoder=predictor.model.image_encoder, out_channels=num_classes, device=device,
    )

    # Load the model weights
    model_state = torch.load(args.checkpoint_path, map_location="cpu")["model_state"]
    unetr.load_state_dict(model_state)
    unetr.to(device)
    unetr.eval()

    pq_per_image = []
    with torch.no_grad():
        for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
            # Read input image and corresponding labels.
            image = read_image(image_path)
            gt = read_image(gt_path)

            # Get the valid region as the remaning is padded
            one_chan = image[:, :, 0]  # Take one channel to extract valid channels.
            idxx = np.argwhere(one_chan > 0)
            x_min, y_min = idxx.min(axis=0)
            x_max, y_max = idxx.max(axis=0)

            image = image[x_min:x_max+1, y_min:y_max+1]
            gt = gt[x_min:x_max+1, y_min:y_max+1]

            # Run inference
            tensor_image = image.transpose(2, 0, 1)
            tensor_image = torch.from_numpy(tensor_image[None]).to(device)
            outputs = unetr(tensor_image)

            # Perform argmax to get per class outputs.
            masks = torch.argmax(outputs, dim=1)
            masks = masks.detach().cpu().numpy().squeeze()

            pq_score = panoptic_quality(gt, masks, class_ids=[1, 2, 3, 4, 5])
            pq_per_image.append(pq_score)

            if args.view:
                # Plot images
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 3, figsize=(30, 15))

                image = image.astype(int)
                ax[0].imshow(image)
                ax[0].axis("off")
                ax[0].set_title("Image", fontsize=20)

                ax[1].imshow(gt)
                ax[1].axis("off")
                ax[1].set_title("Ground Truth", fontsize=20)

                ax[2].imshow(masks)
                ax[2].axis("off")
                ax[2].set_title("Segmentation", fontsize=20)

                plt.savefig("./test.png")
                plt.close()

                breakpoint()

    mpq_neoplastic_cells = np.nanmean([pq[0] for pq in pq_per_image])
    mpq_inflammatory = np.nanmean([pq[1] for pq in pq_per_image])
    mpq_connective = np.nanmean([pq[2] for pq in pq_per_image])
    mpq_dead = np.nanmean([pq[3] for pq in pq_per_image])
    mpq_epithelial = np.nanmean([pq[4] for pq in pq_per_image])

    print(mpq_neoplastic_cells)
    print(mpq_inflammatory)
    print(mpq_connective)
    print(mpq_dead)
    print(mpq_epithelial)
    print()
    print(np.mean(
        [mpq_neoplastic_cells, mpq_inflammatory, mpq_connective, mpq_dead, mpq_epithelial]
    ))


def main(args):
    evaluate_pannuke_semantic_segmentation(args)

    # Results:
    # finetuned_all-from_pretrained
    # 0.5565445094816387
    # 0.263319942526424
    # 0.24964240251909683
    # 0.0
    # 0.5701228149600817
    # Mean: 0.32792593389744823

    # finetuned_all-from_scratch
    # 0.5743457978874937
    # 0.2673637150673506
    # 0.2465035736846288
    # 0.0
    # 0.5860092963168859
    # Mean: 0.3348444765912718

    # finetuned_decoder_only-from_pretrained
    # 0.5354166467281833
    # 0.2439807306507485
    # 0.17926104569693588
    # 0.0
    # 0.5093215049980333
    # Mean: 0.2935959856147802

    # finetuned_decoder_only-from_scratch
    # 0.5211515507951523
    # 0.22803343865046305
    # 0.18502145805422943
    # 0.0
    # 0.4940003394785475
    # Mean: 0.2856413573956784


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/test/data", type=str)
    parser.add_argument("-m", "--model_type", default="vit_b", type=str)
    parser.add_argument("-c", "--checkpoint_path", default=None, type=str)
    parser.add_argument("-v", "--view", default=None, type=str)
    args = parser.parse_args()
    main(args)
