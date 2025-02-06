import os
import collections
from typing import Union, Optional, OrderedDict

import pooch

import torch

from micro_sam.util import microsam_cachedir


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
