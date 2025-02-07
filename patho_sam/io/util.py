import os
from typing import Union, Tuple, Optional

import numpy as np

try:
    import slideio
except ImportError:
    slideio = None


def read_wsi(
    input_path: Union[os.PathLike, str],
    scale: Optional[Tuple[int, int]] = None,
    image_size: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Function to read whole-slide images (WSIs) in histopathology.

    The file formats tested are '.svs', '.scn', '.czi', '.zvi', '.ndpi',
    '.vsi', '.qptiff' and other gdal formats.

    Args:
        input_path: The path to the WSI.
        scale: Relevant for WSIs, to get the image for a desired scale. Provide the desired (H, W) combination to scale.
            You can choose (H, 0) or (0, W) to scale along one dimension and keep the resolution intact.
        image_size: Relevant for WSIs, to get a ROI crop for a desired shape.

    Returns:
        The numpy array.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    assert slideio is not None, "Please install 'slideio': 'pip install slideio'."
    slide = slideio.open_slide(input_path)  # Fetches the slide object.

    # Let's check with expected scale.
    if scale is None:
        scale = (0, 0)  # Loads original resolution.
    else:
        if not isinstance(scale, Tuple) and len(scale) != 2:
            raise ValueError(
                "The scale parameter is expected to be a tuple of height and width dimensions, "
                "such that the new shape is (H', W')"
            )

    # Let's check for the expected size of the desired ROI.
    # NOTE: Here, we expect all values for placing an ROI precisely: (x, y, W, H)
    if image_size is None:
        image_size = (0, 0, 0, 0)
    else:
        if not isinstance(image_size, Tuple):
            raise ValueError(
                "The image size parameter is expected to be a tuple of desired target ROI crop, "
                "such that the new crop shape is for this ROI."
            )

        # If the user provides shapes in the usual 2d axes format, eg. (1024, 1024),
        # we provide them a top-left corner crop.
        if len(image_size) == 2:
            image_size = (0, 0, *image_size)

    assert len(scale) == 2
    assert len(image_size) == 4

    # NOTE: Each slide objects could contain one or multiple scenes,
    # which is coined as a continuous raster region (with the 2d image, other meta-data, etc)
    scene = slide.get_scene(0)
    input_array = scene.read_block(size=scale, rect=image_size)

    return input_array
