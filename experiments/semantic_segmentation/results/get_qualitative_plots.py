import os
from pathlib import Path

import numpy as np
from skimage.segmentation import find_boundaries

from tukra.io import read_image


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/semantic/external"


def get_qualitative_plots():
    # Get the inputs and corresponding labels
    fnames = [
        '1445.tiff', '1145.tiff', '1391.tiff', '2087.tiff',
        '2382.tiff', '0335.tiff', '0594.tiff', '2386.tiff',
        '2316.tiff', '2381.tiff', "1446.tiff", "1407.tiff"
    ]
    image_paths = [os.path.join(ROOT, "semantic_split", "test_images", fname) for fname in fnames]
    gt_paths = [os.path.join(ROOT, "semantic_split", "test_labels", fname) for fname in fnames]

    for image_path, gt_path in zip(image_paths, gt_paths):
        this_fname = os.path.basename(image_path)

        image = read_image(image_path)
        gt = read_image(gt_path)

        # Get an overlay for the input image
        mask_overlay = np.zeros_like(image)
        # Individual colors per class
        mask_overlay[find_boundaries(gt == 1)] = [255, 0, 0]
        mask_overlay[find_boundaries(gt == 2)] = [0, 255, 0]
        mask_overlay[find_boundaries(gt == 3)] = [0, 0, 255]
        mask_overlay[find_boundaries(gt == 4)] = [255, 255, 0]
        mask_overlay[find_boundaries(gt == 5)] = [0, 255, 255]

        # Map overlay over the image
        alpha = 0.5
        overlay = alpha * image + (1.0 - alpha) * mask_overlay
        overlay = overlay.astype("uint8")

        def _get_expected_labels(fpath):
            label = read_image(fpath)

            # Individual colors per class
            label_overlay = np.zeros_like(image)
            label_overlay[label == 1] = [255, 0, 0]
            label_overlay[label == 2] = [0, 255, 0]
            label_overlay[label == 3] = [0, 0, 255]
            label_overlay[label == 4] = [255, 255, 0]
            label_overlay[label == 5] = [0, 255, 255]

            return label_overlay.astype("uint8")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 6, figsize=(30, 20))
        ax[0].imshow(overlay)
        ax[0].axis("off")

        ax[1].imshow(_get_expected_labels(os.path.join(ROOT, "hovernet", "pannuke", this_fname)))
        ax[1].axis("off")

        ax[2].imshow(_get_expected_labels(os.path.join(ROOT,  "hovernext", "pannuke_convnextv2_tiny_1", this_fname)))
        ax[2].axis("off")

        ax[3].imshow(_get_expected_labels(os.path.join(ROOT, "cellvit", "SAM-H-x40", this_fname)))
        ax[3].axis("off")

        ax[4].imshow(_get_expected_labels(f"biomedparse_{this_fname}"))  # I cache predictions on-the-fly atm.
        ax[4].axis("off")

        ax[5].imshow(_get_expected_labels(f"pathosam_{this_fname}"))  # I cache predictions on-the-fly atm.
        ax[5].axis("off")

        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.savefig(("./" + Path(this_fname).stem + ".svg"), bbox_inches="tight")
        plt.close()


def main():
    get_qualitative_plots()


if __name__ == "__main__":
    main()
