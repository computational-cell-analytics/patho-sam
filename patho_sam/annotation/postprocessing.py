import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as cc_label


def _remove_disconnected_components(mask, solidity_threshold, area_threshold) -> np.ndarray:
    """Filter out disconnected components and those below area and solidity thresholds. For disconnected components,
    the component with the largest area is retained while all smaller components are removed."""
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


def postprocess_instance_mask(segmentation: np.ndarray, solidity_threshold: None | float = 0.5,
                              area_threshold: None | int = 25, verbose=False) -> np.ndarray:
    """This function is for postprocessing of patho-sam predictions, especially using APG. It removes
    artifacts like instance pixels disconnected from their main instance or smaller instances entirely enclosed
    by other instances. Additionally, it """
    cleaned = np.zeros_like(segmentation, dtype=np.uint32)

    for prop in tqdm(regionprops(segmentation), disable=not verbose):
        label_id = prop.label
        minr, minc, maxr, maxc = prop.bbox
        roi = segmentation[minr:maxr, minc:maxc]
        mask = (roi == label_id)
        mask = _remove_disconnected_components(mask, solidity_threshold, area_threshold)
        filled = binary_fill_holes(mask)

        cleaned[minr:maxr, minc:maxc] = np.where(filled, label_id, cleaned[minr:maxr, minc:maxc])

    return cleaned
