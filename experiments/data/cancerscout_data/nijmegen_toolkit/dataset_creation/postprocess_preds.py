import os
import argparse
import imageio.v3 as imageio
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
from glob import glob
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as cc_label

NIJMEGEN_ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/ignite_nijmegen"


def remove_disconnected_components(mask, solidity_threshold, area_threshold):
    """Filter out disconnected components and those below area and solidity thresholds"""
    structure = np.ones((3, 3), dtype=int)
    labeled_cc, num_components = cc_label(mask, structure=structure)
    filtered_mask = np.zeros_like(mask, dtype=bool)
    props = regionprops(labeled_cc)

    if num_components > 1:
        areas = [p.area for p in props]
        largest_idx = np.argmax(areas)
        instance_label = props[largest_idx].label
        instance_area = props[largest_idx].area
        instance_solidity = props[largest_idx].solidity

    else:
        instance_label = props[0].label
        instance_area = props[0].area
        instance_solidity = props[0].solidity

    if instance_area > area_threshold and instance_solidity > solidity_threshold:
        filtered_mask[labeled_cc == instance_label] = True

    return filtered_mask


def remove_disconnected_components_and_fill_holes(label_img, solidity_threshold, area_threshold):
    cleaned = np.zeros_like(label_img, dtype=np.uint32)

    for prop in tqdm(regionprops(label_img)):
        label_id = prop.label
        minr, minc, maxr, maxc = prop.bbox
        roi = label_img[minr:maxr, minc:maxc]
        mask = (roi == label_id)
        mask = remove_disconnected_components(mask, solidity_threshold, area_threshold)
        filled = binary_fill_holes(mask)

        cleaned[minr:maxr, minc:maxc] = np.where(filled, label_id, cleaned[minr:maxr, minc:maxc])

    return cleaned


def postprocess_predictions(input_root, solidity_threshold, area_threshold):
    pred_paths = glob(os.path.join(input_root, "predictions", "*"))
    filtered_dir = os.path.join(input_root, "filtered_predictions_v5")
    os.makedirs(filtered_dir, exist_ok=True)

    for pred_path in tqdm(pred_paths):
        pred_out_path = os.path.join(filtered_dir, os.path.basename(pred_path))
        pred = imageio.imread(pred_path)
        pred = remove_disconnected_components_and_fill_holes(pred, solidity_threshold, area_threshold)

        imageio.imwrite(pred_out_path, pred, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default=NIJMEGEN_ROOT)
    parser.add_argument("--solidity_threshold", type=float, default=0.5)
    parser.add_argument("--area_threshold", type=int, default=35)
    args = parser.parse_args()
    postprocess_predictions(
        input_root=args.input_root,
        solidity_threshold=args.solidity_threshold,
        area_threshold=args.area_threshold,
        )


if __name__ == "__main__":
    main()
