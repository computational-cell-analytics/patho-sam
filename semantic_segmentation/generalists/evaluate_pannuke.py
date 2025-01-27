import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np
import pandas as pd

import torch

from tukra.io import read_image

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_unetr

from patho_sam.evaluation import semantic_segmentation_quality


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

    sq_per_image = []
    with torch.no_grad():
        for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
            # Read input image and corresponding labels.
            image = read_image(image_path)
            gt = read_image(gt_path)

            # Run inference
            tensor_image = image.transpose(2, 0, 1)
            tensor_image = torch.from_numpy(tensor_image[None]).to(device)
            outputs = unetr(tensor_image)

            # Perform argmax to get per class outputs.
            masks = torch.argmax(outputs, dim=1)
            masks = masks.detach().cpu().numpy().squeeze()

            # Get the valid region as the remaining is padded.
            one_chan = image[:, :, 0]  # Take one channel to extract valid channels.
            idxx = np.argwhere(one_chan > 0)
            x_min, y_min = idxx.min(axis=0)
            x_max, y_max = idxx.max(axis=0)

            # Crop out valid region in image and corresponding labels and segmentation.
            image = image[x_min:x_max+1, y_min:y_max+1]
            gt = gt[x_min:x_max+1, y_min:y_max+1]
            masks = masks[x_min:x_max+1, y_min:y_max+1]

            # Calcuate the score.
            sq_score = semantic_segmentation_quality(gt, masks, class_ids=[1, 2, 3, 4, 5])
            sq_per_image.append(sq_score)

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

    msq_neoplastic_cells = np.nanmean([sq[0] for sq in sq_per_image])
    msq_inflammatory = np.nanmean([sq[1] for sq in sq_per_image])
    msq_connective = np.nanmean([sq[2] for sq in sq_per_image])
    msq_dead = np.nanmean([sq[3] for sq in sq_per_image])
    msq_epithelial = np.nanmean([sq[4] for sq in sq_per_image])

    results = {
        "neoplastic_cells": msq_neoplastic_cells,
        "inflammatory_cells": msq_inflammatory,
        "connective_cells": msq_connective,
        "dead_cells": msq_dead,
        "epithelial_cells": msq_epithelial,
        "mean": np.mean([msq_neoplastic_cells, msq_inflammatory, msq_connective, msq_dead, msq_epithelial]),
    }
    results = pd.DataFrame.from_dict([results])
    print(results)


def main(args):
    evaluate_pannuke_semantic_segmentation(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/test/data", type=str)
    parser.add_argument("-m", "--model_type", default="vit_b", type=str)
    parser.add_argument("-c", "--checkpoint_path", required=True)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    main(args)
