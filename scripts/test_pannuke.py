import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import h5py
import numpy as np
import imageio.v3 as imageio

from torch_em.data.datasets.histopathology import pannuke

from micro_sam.util import get_sam_model
from micro_sam.evaluation import inference, evaluation


def _pad_image(image, target_shape=(512, 512), pad_value=0):
    pad = [
        (max(t - s, 0) // 2, max(t - s, 0) - max(t - s, 0) // 2) for s, t in zip(image.shape[:2], target_shape)
    ]

    if image.ndim == 3:
        pad.append((0, 0))

    return np.pad(image, pad, mode="constant", constant_values=pad_value)


def run_interactive_segmentation(input_path, experiment_folder, model_type, start_with_box_prompt=True):

    # Create clone of single images in input_path directory.
    data_dir = os.path.join(input_path, "benchmark_2d")

    if not os.path.exists(data_dir):
        # First, we get the fold 3.
        fold_path = pannuke.get_pannuke_paths(path=input_path, folds=["fold_3"], download=True)
        fold_path = fold_path[0]

        # Next, simply extract the images.
        with h5py.File(fold_path, "r") as f:
            image_stack = f["images"][:].transpose(1, 2, 3, 0)
            label_stack = f["labels/instances"][:]

        # Store them one-by-one locally in an experiment folder.
        os.makedirs(data_dir)

        for i, (image, label) in tqdm(
            enumerate(zip(image_stack, label_stack)), total=len(image_stack), desc="Extracting images",
        ):
            # There has to be some foreground in the image to be considered for interactive segmentation.
            if len(np.unique(label)) == 1:
                continue

            image = _pad_image(image)
            label = _pad_image(label)

            imageio.imwrite(os.path.join(data_dir, f"pannuke_fold_3_{i:05}_image.tif"), image, compression="zlib")
            imageio.imwrite(os.path.join(data_dir, f"pannuke_fold_3_{i:05}_label.tif"), label, compression="zlib")

    # Well, now we have our image and label paths.
    image_paths = natsorted(glob(os.path.join(data_dir, "*_image.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*_label.tif")))

    assert len(image_paths) == len(label_paths)

    # Now that we have the data ready, run interactive segmentation.

    # HACK: for debugging purpose, I will check on first 100 images. Remove the next line to run it on all images.
    image_paths, label_paths = image_paths[:100], label_paths[:100]

    # Get the Segment Anything model.
    predictor = get_sam_model(model_type=model_type)

    # Then run interactive segmentation by simulating prompts from labels.
    prediction_root = os.path.join(
        experiment_folder, ("start_with_box" if start_with_box_prompt else "start_with_point")
    )
    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=label_paths,
        embedding_dir=None,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt,
    )

    # And evaluate the results.
    results = evaluation.run_evaluation_for_iterative_prompting(
        gt_paths=label_paths,
        prediction_root=prediction_root,
        experiment_folder=experiment_folder,
        start_with_box_prompt=start_with_box_prompt,
    )

    print(results)


def main(args):
    run_interactive_segmentation(
        input_path=args.input_path,
        model_type=args.model_type,
        experiment_folder=args.experiment_folder,
        start_with_box_prompt=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/data/pannuke", type=str)
    parser.add_argument("-e", "--experiment_folder", default="./experiments", type=str)
    parser.add_argument("-m", "--model_type", default="vit_b_histopathology", type=str)
    args = parser.parse_args()
    main(args)
