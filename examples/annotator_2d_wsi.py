import os

from micro_sam.util import get_cache_directory
from micro_sam.sam_annotator import annotator_2d
from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data

from patho_sam.io import read_wsi


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def whole_slide_image_annotator(use_finetuned_model):
    """Run the 2d annotator for an example whole-slide image for histopathology.

    See 'fetch_wholeslide_histopathology_example_data' for details on the data.
    """
    example_data = fetch_wholeslide_histopathology_example_data(DATA_CACHE)

    # Load the WSI image.
    image = read_wsi(example_data)

    if use_finetuned_model:
        model_type = "vit_b_histopathology"
    else:
        model_type = "vit_b"

    # Store embeddings in a desired shape.
    save_path = f"./embedding_{model_type}.zarr"

    annotator_2d(
        image=image,
        embedding_path=save_path,
        model_type=model_type,
        tile_shape=(384, 384),
        halo=(64, 64),
        precompute_amg_state=True,
    )


def main():
    # Whether to use the fine-tuned SAM model for WSIs.
    use_finetuned_model = True

    # 2d annotator for WSI data.
    whole_slide_image_annotator(use_finetuned_model)


if __name__ == "__main__":
    main()
