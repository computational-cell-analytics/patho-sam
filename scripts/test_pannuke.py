import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import h5py
import numpy as np
import imageio.v3 as imageio
from skimage.segmentation import relabel_sequential

from torch_em.data.datasets.histopathology import pannuke, monuseg

from micro_sam.util import get_sam_model
from micro_sam.evaluation import inference, evaluation


def _pad_image(image, target_shape=(512, 512), pad_value=0):
    pad = [
        (max(t - s, 0) // 2, max(t - s, 0) - max(t - s, 0) // 2) for s, t in zip(image.shape[:2], target_shape)
    ]

    if image.ndim == 3:
        pad.append((0, 0))

    return np.pad(image, pad, mode="constant", constant_values=pad_value)


def get_data_paths(input_path, dataset_name):
    # Set specific data folders.
    input_path = os.path.join(input_path, dataset_name)

    if dataset_name == "pannuke":
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

                # NOTE: I am padding the image below to match the shape of inputs on which it is trained,
                # i.e. (512, 512), for proper reproducibility (otherwise the results are slightly worse)
                image = _pad_image(image)
                label = _pad_image(label)

                imageio.imwrite(os.path.join(data_dir, f"pannuke_fold_3_{i:05}_image.tif"), image, compression="zlib")
                imageio.imwrite(os.path.join(data_dir, f"pannuke_fold_3_{i:05}_label.tif"), label, compression="zlib")

        # Well, now we have our image and label paths.
        image_paths = natsorted(glob(os.path.join(data_dir, "*_image.tif")))
        label_paths = natsorted(glob(os.path.join(data_dir, "*_label.tif")))

        assert len(image_paths) == len(label_paths)

        # HACK: for debugging purpose, I will check on first 100 images. Remove the next line to run it on all images.
        image_paths, label_paths = image_paths[:100], label_paths[:100]

    elif dataset_name == "monuseg":
        # Create clone of cropped images in input_path directory.
        data_dir = os.path.join(input_path, "benchmark_2d")
        os.makedirs(data_dir, exist_ok=True)

        curr_image_paths, curr_label_paths = monuseg.get_monuseg_paths(path=input_path, split="test", download=True)

        # Let's do a simple cropping to test stuff.
        image_paths, label_paths = [], []
        for curr_image_path, curr_label_path in zip(curr_image_paths, curr_label_paths):
            image = imageio.imread(curr_image_path)
            label = imageio.imread(curr_label_path).astype("uint32")

            # Do a simple cropping and relabel instances.
            image, label = image[:512, :512, :], label[:512, :512]
            label = relabel_sequential(label)[0]

            # And save the cropped image and corresponding label.
            image_path = os.path.join(data_dir, os.path.basename(curr_image_path))
            image_paths.append(image_path)
            imageio.imwrite(image_path, image, compression="zlib")

            label_path = os.path.join(data_dir, os.path.basename(curr_label_path))
            label_paths.append(label_path)
            imageio.imwrite(label_path, label, compression="zlib")

    else:
        raise ValueError

    return image_paths, label_paths


def run_interactive_segmentation(input_path, experiment_folder, model_type, start_with_box_prompt=True):

    # Setup 1: PanNuke images (pad (256, 256) images up to (512, 512) to match the training patch shape)
    # image_paths, label_paths = get_data_paths(input_path, "pannuke")  # NOTE: uncomment to run it on PanNuke

    # Setup 2: MoNuSeg images (since the images are larger than training patch shape, we crop them to shape (512, 512))
    image_paths, label_paths = get_data_paths(input_path, "monuseg")  # NOTE: comment this before running other setups.

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
        start_with_box_prompt=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/data", type=str)
    parser.add_argument("-e", "--experiment_folder", default="./experiments", type=str)
    parser.add_argument("-m", "--model_type", default="vit_b_histopathology", type=str)
    args = parser.parse_args()
    main(args)
