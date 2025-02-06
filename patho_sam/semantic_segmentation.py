import os

from typing import Optional, Union, Tuple

import numpy as np

from nifty.tools import blocking

import torch

from micro_sam import util
from micro_sam.instance_segmentation import get_unetr, DecoderAdapter, _process_tiled_embeddings

from segment_anything.predictor import SamPredictor

from .util import get_semantic_segmentation_decoder_weights


def get_semantic_predictor_and_segmenter(
    model_type: str,
    checkpoint: Optional[Union[os.PathLike, str]] = None,
    device: str = None,
    is_tiled: bool = False,
    num_classes: int = 6,  # Default set to PanNuke's total number of classes (and additional class count for bg)
):
    """Get the Segment Anything for Histopathology model and class for automatic semantic segmentation.

    Args:
        model_type: The PathoSAM model choice.
        checkpoint: The filepath to the stored model checkpoints.
        device: The torch device.
        is_tiled: Whether to return segmenter for performing segmentation in tiling window style.
        num_classes: The total number of classes for semantic segmentation.

    Returns:
        The Segment Anything model.
        The automatic semantic segmentation class.
    """

    # NOTE: Support is limited to 'vit_b_histopathology' model, as we provide the pretrained decoder only for this.
    if model_type != "vit_b_histopathology":
        raise RuntimeError(
            "It is only possible to run semantic segmentation with pretrained decoder for 'vit_b_histopathology' model."
        )

    # Get the device
    device = util.get_device(device=device)

    # Ensure that the users use our PathoSAM models for this task.
    if checkpoint is None and not model_type.endswith("_histopathology"):
        raise RuntimeError("Please choose the PathoSAM generalist models.")

    # Get the predictor for PathoSAM models.
    predictor = util.get_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint)

    # Downloads the decoder state automatically and allows loading it.
    decoder_state = get_semantic_segmentation_decoder_weights()

    # Get the decoder for semantic segmentation.
    decoder = DecoderAdapter(
        unetr=get_unetr(
            image_encoder=predictor.model.image_encoder,
            decoder_state=decoder_state,
            device=device,
            out_channels=num_classes,
        )
    )

    segmenter_class = TiledSemanticSegmentationWithDecoder if is_tiled else SemanticSegmentationWithDecoder
    segmenter = segmenter_class(predictor=predictor, decoder=decoder)

    return predictor, segmenter


class SemanticSegmentationWithDecoder:
    """Generates a semantic segmentation without prompts, using a decoder.

    Implements the same interace from `micro-sam` as `InstanceSegmentationWithDecoder`.

    Use this class as follows:
    ```python
    segmenter = SemanticSegmentationWithDecoder(predictor, decoder)
    segmenter.initialize(image)  # Predicts the image embeddings and decoder outputs.
    masks = segmenter.generate()  # Generate the semantic segmentation.
    ```

    Args:
        predictor: The segment anything predictor.
        decoder: The decoder to predict per-class outputs for semantic segmentation.
    """
    def __init__(self, predictor: SamPredictor, decoder: torch.nn.Module) -> None:
        self._predictor = predictor
        self._decoder = decoder

        # The decoder output.
        self._semantic_segmentation = None

        self._is_initialized = False

    @property
    def is_initialized(self):
        """Whether the mask generator has already been initialized.
        """
        return self._is_initialized

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        image_embeddings: Optional[util.ImageEmbeddings] = None,
        num_classes: int = 6,  # Default set to PanNuke's total number of classes (and additional class count for bg)
        i: Optional[int] = None,
        verbose: bool = False,
        pbar_init: Optional[callable] = None,
        pbar_update: Optional[callable] = None,
    ) -> None:
        """Initialize image embeddings and decoder predictions for an image.

        Args:
            image: The input image, volume or timeseries.
            image_embeddings: Optional precomputed image embeddings.
                See `util.precompute_image_embeddings` for details.
            num_classes: The total number of classes for semantic segmentation.
            i: Index for the image data. Required if `image` has three spatial dimensions
                or a time dimension and two spatial dimensions.
            verbose: Whether to be verbose.
            pbar_init: Callback to initialize an external progress bar. Must accept number of steps and description.
                Can be used together with pbar_update to handle napari progress bar in other thread.
                To enables using this function within a threadworker.
            pbar_update: Callback to update an external progress bar.
        """
        _, pbar_init, pbar_update, pbar_close = util.handle_pbar(verbose, pbar_init, pbar_update)
        pbar_init(1, "Initialize semantic segmentation with decoder")

        if image_embeddings is None:
            image_embeddings = util.precompute_image_embeddings(predictor=self._predictor, input_=image, ndim=2)

        # Get the image embeddings from the predictor.
        self._predictor = util.set_precomputed(self._predictor, image_embeddings, i=i)
        embeddings = self._predictor.features
        input_shape = tuple(self._predictor.input_size)
        original_shape = tuple(self._predictor.original_size)

        # Run prediction with the UNETR decoder.
        output = self._decoder(embeddings, input_shape, original_shape)

        assert output.shape[1] == num_classes, f"{output.shape}"

        # Get the per-class outputs into one valid mask.
        output = torch.argmax(output, dim=1)
        output = output.detach().cpu().numpy().squeeze()

        pbar_update(1)
        pbar_close()

        # Set the state.
        self._semantic_segmentation = output
        self._is_initialized = True

    def generate(self) -> np.ndarray:
        """Generate semantic segmentation for the currently initialized image.

        Returns:
            The semantic segmentation masks.
        """
        if not self.is_initialized:
            raise RuntimeError("SemanticSegmentationWithDecoder has not been initialized. Call initialize first.")

        return self._semantic_segmentation


class TiledSemanticSegmentationWithDecoder(SemanticSegmentationWithDecoder):
    """Same as `SemanticSegmentationWithDecoder` but for tiled image embeddings.
    """

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        image_embeddings: Optional[util.ImageEmbeddings] = None,
        num_classes: int = 6,  # Default set to PanNuke's total number of classes (and additional class count for bg)
        i: Optional[int] = None,
        tile_shape: Optional[Tuple[int, int]] = None,
        halo: Optional[Tuple[int, int]] = None,
        verbose: bool = False,
        pbar_init: Optional[callable] = None,
        pbar_update: Optional[callable] = None,
    ) -> None:
        """Initialize image embeddings and decoder predictions for an image.

        Args:
            image: The input image, volume or timeseries.
            image_embeddings: Optional precomputed image embeddings.
                See `util.precompute_image_embeddings` for details.
            i: Index for the image data. Required if `image` has three spatial dimensions
                or a time dimension and two spatial dimensions.
            tile_shape: Shape of the tiles for precomputing image embeddings.
            halo: Overlap of the tiles for tiled precomputation of image embeddings.
            verbose: Dummy input to be compatible with other function signatures.
            pbar_init: Callback to initialize an external progress bar. Must accept number of steps and description.
                Can be used together with pbar_update to handle napari progress bar in other thread.
                To enables using this function within a threadworker.
            pbar_update: Callback to update an external progress bar.
        """
        original_size = image.shape[:2]
        image_embeddings, tile_shape, halo = _process_tiled_embeddings(
            self._predictor, image, image_embeddings, tile_shape, halo, verbose=verbose,
        )
        tiling = blocking([0, 0], original_size, tile_shape)

        _, pbar_init, pbar_update, pbar_close = util.handle_pbar(verbose, pbar_init, pbar_update)
        pbar_init(tiling.numberOfBlocks, "Initialize tiled semantic segmentation with decoder")

        semantic_segmentation = np.zeros(original_size, dtype="uint8")

        for tile_id in range(tiling.numberOfBlocks):

            # Get the image embeddings from the predictor for this tile.
            self._predictor = util.set_precomputed(self._predictor, image_embeddings, i=i, tile_id=tile_id)
            embeddings = self._predictor.features
            input_shape = tuple(self._predictor.input_size)
            original_shape = tuple(self._predictor.original_size)

            # Run prediction with the UNETR decoder.
            output = self._decoder(embeddings, input_shape, original_shape)

            assert output.shape[1] == num_classes, f"{output.shape}"

            # Get the per-class outputs into one valid mask.
            output = torch.argmax(output, dim=1)
            output = output.detach().cpu().numpy().squeeze()

            # Set the predictions in the output for this tile.
            block = tiling.getBlockWithHalo(tile_id, halo=list(halo))
            local_bb = tuple(
                slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end)
            )
            inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))

            semantic_segmentation[inner_bb] = output[local_bb]
            pbar_update(1)

        pbar_close()

        # Set the state.
        self._semantic_segmentation = semantic_segmentation
        self._is_initialized = True
