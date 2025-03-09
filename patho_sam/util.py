import os
import collections
from typing import Union, Optional, OrderedDict

import pooch

import torch

from micro_sam.util import microsam_cachedir, get_cache_directory
from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data

from .io import read_wsi


DECODER_URL = "https://owncloud.gwdg.de/index.php/s/TFbI25UZoixd1hi/download"


def export_semantic_segmentation_decoder(
    checkpoint_path: Union[str, os.PathLike], save_path: Union[str, os.PathLike],
):
    """Exports the weights of the trained convolutional decoder for semantic segemntation task.

    Args:
        checkpoint_path: Filepath to the trained semantic segmentation checkpoint.
        save_path: Filepath where the decoder weights will be stored.
    """
    # Load the model state from finetuned checkpoint.
    model_state = torch.load(checkpoint_path, map_location="cpu")["model_state"]

    # Get the decoder state only.
    decoder_state = collections.OrderedDict(
        [(k, v) for k, v in model_state.items() if not k.startswith("encoder")]
    )

    # Store the decoder state to a desired path.
    torch.save(decoder_state, save_path)


def get_semantic_segmentation_decoder_weights(save_path: Optional[Union[str, os.PathLike]] = None) -> OrderedDict:
    """Get the semantic segmentation decoder weights for initializing the decoder-only.

    Args:
        save_path: Whether to save the model checkpoints to desired path.

    Returns:
        The pretrained decoder weights.
    """
    # By default, we store decoder weights to `micro-sam` cache directory.
    save_directory = os.path.join(microsam_cachedir(), "models") if save_path is None else save_path

    # Download the model weights
    fname = "vit_b_histopathology_semantic_segmentation_decoder"
    pooch.retrieve(
        url=DECODER_URL,
        known_hash="bdd05a55c72c02abce72a7aa6885c6ec21df9c43fda9cf3c5d11ef5788de0ab0",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )

    # Get the checkpoint path.
    checkpoint_path = os.path.join(save_directory, fname)

    # Load the decoder state.
    state = torch.load(checkpoint_path, map_location="cpu")

    return state


def get_example_wsi_data():
    """@private"""
    import argparse
    parser = argparse.ArgumentParser(description="Download and visualize the example whole-slide image (WSI).")
    parser.add_argument(
        "-s", "--save_path", type=str, default=None,
        help=f"The folder to store the whole-slide image. By default, it is stored at '{get_cache_directory()}'."
    )
    parser.add_argument(
        "--roi", nargs="+", type=int, default=None,
        help="The roi shape of the whole slide image for automatic segmentation. By default, predicts on entire WSI. "
        "You can provide the ROI shape as: '--roi X Y W H'.",
    )
    parser.add_argument(
        "--view", action="store_true", help="Whether to view the WSI in napari."
    )

    args = parser.parse_args()

    # Get the folder to store the WSI. By default, stores it at 'micro-sam' cache directory.
    save_dir = os.path.join(get_cache_directory(), "sample_data") if args.save_path is None else args.save_dir

    # Download the example WSI.
    example_data = fetch_wholeslide_histopathology_example_data(save_dir)

    if args.view:
        # Load the WSI image.
        image = read_wsi(example_data, image_size=args.roi)

        # Get multi-scales for the input image.
        multiscale_images = [
            image,
            read_wsi(example_data, image_size=args.roi, scale=(int(image.shape[0] / 2), 0)),
            read_wsi(example_data, image_size=args.roi, scale=(int(image.shape[0] / 4), 0)),
            read_wsi(example_data, image_size=args.roi, scale=(int(image.shape[0] / 8), 0)),
        ]

        import napari
        v = napari.Viewer()
        v.add_image(multiscale_images, name="Input Image")
        napari.run()

    print(f"The example WSI is stored at '{example_data}'.")
