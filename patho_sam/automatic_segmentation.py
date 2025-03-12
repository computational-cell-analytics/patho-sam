import os
import time
from pathlib import Path
from typing import Optional, Union, Literal, Tuple

import numpy as np
import imageio.v3 as imageio
from skimage.transform import resize

import torch

from micro_sam.util import precompute_image_embeddings
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from .io import read_wsi
from .semantic_segmentation import get_semantic_predictor_and_segmenter


def _add_suffix_to_output_path(output_path: Union[str, os.PathLike], suffix: str) -> str:
    fpath = Path(output_path).resolve()
    fext = fpath.suffix if fpath.suffix else ".tif"
    return str(fpath.with_name(f"{fpath.stem}{suffix}{fext}"))


def automatic_segmentation_wsi(
    input_image: Union[np.ndarray, str, os.PathLike],
    model_type: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    output_path: Optional[Union[str, os.PathLike]] = None,
    tile_shape: Tuple[int, int] = (384, 384),
    halo: Tuple[int, int] = (64, 64),
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    device: Optional[Union[str, torch.device]] = None,
    output_choice: Literal["instances", "semantic", "all"] = "instances",
    batch_size: int = 1,
    verbose: bool = False,
    view: bool = False,
) -> np.ndarray:
    """Run automatic segmentation for a whole-slide input image.

    Args:
        input_image: The whole-slide image.
        model_type: The PathoSAM model.
        output_path: The filepath where the segmentations will be stored.
        tile_shape: The tile shape for tiling-window prediction.
        halo: The overlap for tiling-window prediction.
        checkpoint_path: The filepath for Segment Anything model checkpoints.
        embedding_path: The filepath where the precomputed embeddings are cached.
        device: The device to run automatic segmentation on.
        output_choice: The choice of outputs. Either 'instances' / 'semantic' / 'all'.
        batch_size: The batch size to compute image embeddings over tiles.
        verbose: Whether to allow verbosity of all processes.
        view: Whether to visualize the segmentations in napari.

    Returns:
        The segmentation result.
    """
    if output_choice not in ["instances", "semantic", "all"]:
        raise ValueError(
            f"'{output_choice}' is not a supported output choice. Choose either 'instances' / 'semantic' / 'all'."
        )

    # Set initial values.
    instance_masks = None
    semantic_masks = None
    image_embeddings = None

    # Ensure provided arguments are in expected format.
    if tile_shape and not isinstance(tile_shape, tuple):
        tile_shape = tuple(tile_shape)
    if halo and not isinstance(halo, tuple):
        halo = tuple(halo)
    if roi and not isinstance(roi, tuple):
        roi = tuple(roi)

    # Get additional suffix for ROIs to store results with ROI values.
    if roi:
        if len(roi) != 4:
            raise RuntimeError("The provided ROI is not valid. Please provide the ROI shape as: '--roi X Y W H'.")

        roi_suffix = f"_ROI_X{roi[0]}-{roi[0] + roi[2]}_Y{roi[1]}-{roi[1] + roi[3]}"
    else:
        roi_suffix = ""

    # Ensure that the user makes use of AIS, i.e. the most favorable automatic segmentation method.
    if checkpoint_path is None and not model_type.endswith("_histopathology"):
        raise RuntimeError("Please choose the PathoSAM generalist models.")

    # Read the WSI image
    if isinstance(input_image, Union[str, os.PathLike]):  # from a filepath.
        image = read_wsi(input_image, image_size=roi)
    else:  # or use the input array as it is.
        image = input_image

    # 1. Run automatic instance segmentation.
    if output_choice != "semantic":  # do instance segmentation always besides "semantic"-only as 'output_choice'.
        instances_save_path = _add_suffix_to_output_path(output_path, roi_suffix + "_instances")
        # Run instance segmentation only if it is not saved already.
        if os.path.exists(instances_save_path):
            instance_masks = imageio.imread(instances_save_path)
            print(f"The instance segmentation results are already stored at '{instances_save_path}'.")
        else:
            # Get the predictor and segmenter for automatic instance segmentation.
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model_type,
                checkpoint=checkpoint_path,
                device=device,
                amg=False,  # i.e. run AIS.
                is_tiled=True,  # i.e. run tiling-window based segmentation.
            )

            instance_masks, image_embeddings = automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmenter,
                input_path=image,
                output_path=instances_save_path,  # Stores instance segmentation.
                embedding_path=embedding_path,
                ndim=2,  # We hard-code this as whole-slide images are ideally always 2d images in RGB-style.
                tile_shape=tile_shape,
                halo=halo,
                verbose=verbose,
                output_mode=None,  # Skips some post-processing under `generate` method after automatic seg.
                return_embeddings=True,  # Returns image embeddings, can be used in the task below, i.e. semantic seg.
                batch_size=batch_size,
            )
            print("The instance segmentation results have been computed.")

    # 2. Run semantic segmentation.
    if output_choice != "instances":  # do semantic segmentation always besides "instances"-only as 'output_choice'.
        semantic_save_path = _add_suffix_to_output_path(output_path, roi_suffix + "_semantic")
        # Run semantic segmentation only if it is not saved already.
        if os.path.exists(semantic_save_path):
            semantic_masks = imageio.imread(semantic_save_path)
            print(f"The semantic segmentation results are already stored at '{semantic_save_path}'.")
        else:
            # Get the predictor and segmenter for automatic semantic segmentation.
            predictor, segmenter = get_semantic_predictor_and_segmenter(
                model_type=model_type, checkpoint=checkpoint_path, device=device, is_tiled=True,
            )

            if image_embeddings is None:
                # Precompute the image embeddings.
                image_embeddings = precompute_image_embeddings(
                    predictor=predictor,
                    input_=image,
                    save_path=embedding_path,
                    ndim=2,
                    tile_shape=tile_shape,
                    halo=halo,
                    verbose=verbose,
                    batch_size=batch_size,
                )

            segmenter.initialize(
                image=image, image_embeddings=image_embeddings, tile_shape=tile_shape, halo=halo, verbose=verbose,
            )
            semantic_masks = segmenter.generate()

            # Store the results.
            imageio.imwrite(semantic_save_path, semantic_masks, compression="zlib")
            print("The semantic segmentation results have been computed.")

    # Store all possible segmentations in the desired output filepath.
    segmentations = []
    if instance_masks is not None:
        segmentations.append(instance_masks)
    if semantic_masks is not None:
        segmentations.append(semantic_masks)

    segmentations = np.stack(segmentations, axis=0).squeeze()
    imageio.imwrite(_add_suffix_to_output_path(output_path, roi_suffix), segmentations, compression="zlib")

    if view:
        # Get multi-scales for the input image.
        multiscale_images = [
            image,
            read_wsi(input_image, image_size=roi, scale=(int(image.shape[0] / 2), 0)),
            read_wsi(input_image, image_size=roi, scale=(int(image.shape[0] / 4), 0)),
            read_wsi(input_image, image_size=roi, scale=(int(image.shape[0] / 8), 0)),
        ]

        # Enable multi-scale for labels for smoother transition.
        def _get_multiscale_labels(original_masks):
            multiscale_labels = [original_masks]
            for im in multiscale_images[1:]:
                ds_labels = resize(
                    multiscale_labels[-1], im.shape[:2], order=0, preserve_range=True, anti_aliasing=False,
                ).astype(original_masks.dtype)
                multiscale_labels.append(ds_labels)
            return multiscale_labels

        import napari
        v = napari.Viewer()

        v.add_image(multiscale_images, name="Input Image")
        if instance_masks is not None:
            multiscale_instance_masks = _get_multiscale_labels(instance_masks)
            v.add_labels(multiscale_instance_masks, name="Instance Segmentation")
        if semantic_masks is not None:
            multiscale_semantic_masks = _get_multiscale_labels(semantic_masks)
            v.add_labels(multiscale_semantic_masks, name="Semantic Segmentation")

        napari.run()

    return segmentations


def main():
    """@private"""
    import argparse

    available_models = ["vit_b_histopathology", "vit_l_histopathology", "vit_h_histopathology"]
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(description="Run automatic segmentation for a whole-slide image (WSI).")
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="The filepath to image data. Supports all data types that can be read by imageio (eg. 'tif', 'png', ...) "
        "or slideio (eg. 'svs', 'scn', 'czi', 'zvi', 'ndpi', 'vsi', 'qptiff' and other 'gdal' formats)."
    )
    parser.add_argument(
        "--roi", nargs="+", type=int, default=None,
        help="The roi shape of the whole slide image for automatic segmentation. By default, predicts on entire WSI. "
        "You can provide the ROI shape as: '--roi X Y W H'.",
    )
    parser.add_argument(
        "-o", "--output_path", required=True,
        help="The filepath to store the automatic segmentation. The current support stores segmentation in 'tif' file. "
        "In addition, the respective automatic segmentation outputs are stored in individual files as well. "
        "eg. instance segmentation is stored at <FILENAME>_instances.tif, and semantic at <FILENAME>_semantic.tif. "
    )
    parser.add_argument(
        "-e", "--embedding_path", default=None, type=str, help="The path where the embeddings will be saved."
    )
    parser.add_argument(
        "-m", "--model_type", default="vit_b_histopathology",
        help=f"The segment anything model that will be used, one of {available_models}."
    )
    parser.add_argument(
        "-c", "--checkpoint", default=None, help="Checkpoint from which the SAM model will be loaded."
    )
    parser.add_argument(
        "--tile_shape", nargs="+", type=int, default=(384, 384),
        help="The tile shape for using tiled prediction. You can provide the tile shape as: '--tile_shape 384 384'.",
    )
    parser.add_argument(
        "--halo", nargs="+", type=int, default=(64, 64),
        help="The overlap shape for using tiled prediction. You can provide the overlap shape as: '--halo 64 64'.",
    )
    parser.add_argument(
        "--output_choice", type=str, default="instances",
        help="The choice of automatic segmentation with the PathoSAM models. Either 'instances' / 'semantic' / 'all'."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to allow verbosity of outputs."
    )
    parser.add_argument(
        "--view", action="store_true", help="Whether to view the segmentations in napari."
    )
    parser.add_argument(
        "-d", "--device", default=None,
        help="The device to use for the predictor. Can be one of 'cuda', 'cpu' or 'mps' (only MAC)."
        "By default the most performant available device will be selected."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="The batch size for computing image embeddings with tiling. By default, set to 1."
    )

    args = parser.parse_args()

    # Start timing the automatic segmentation process.
    start_time = time.time()

    automatic_segmentation_wsi(
        input_image=args.input_path,
        model_type=args.model_type,
        roi=args.roi,
        output_path=args.output_path,
        tile_shape=args.tile_shape,
        halo=args.halo,
        checkpoint_path=args.checkpoint,
        embedding_path=args.embedding_path,
        device=args.device,
        output_choice=args.output_choice,
        verbose=args.verbose,
        view=args.view,
        batch_size=args.batch_size,
    )

    # Calculate the end time of the process.
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Get the times in hour:min:sec format.
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"The automatic segmentation took: {hours:02d}:{minutes:02d}:{seconds:02d} hours.")


if __name__ == "__main__":
    main()
