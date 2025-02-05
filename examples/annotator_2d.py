import os

from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data

from patho_sam.io import read_wsi


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def whole_slide_image_annotator(use_finetuned_model):
    """Run the 2d annotator for an example whole-slide image for histopathology.

    See 'fetch_wholeslide_histopathology_example_data' for details on the data.
    """
    example_data = fetch_wholeslide_histopathology_example_data(DATA_CACHE)

    # image = read_wsi(example_data, image_size=(10000, 15000, 5000, 5000), scale=None)  # working ROI
    # image = read_wsi(example_data, image_size=(10000, 10000, 10000, 10000), scale=None)  # decent shaped ROI
    image = read_wsi(example_data, image_size=None, scale=None)  # full slide shape: (32914, 46000, 3)

    breakpoint()

    if use_finetuned_model:
        model_type = "vit_b_histopathology"
    else:
        model_type = "vit_b"

    # Store embeddings in a desired shape.
    save_path = f"./embedding_{model_type}.zarr"

    automatic_segmentation = True
    if automatic_segmentation:
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
        predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, amg=False, is_tiled=True)
        instances = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image,
            output_path="./test.tif",
            embedding_path=save_path,
            ndim=2,
            tile_shape=(384, 384),
            halo=(64, 64),
        )

    else:
        from micro_sam.sam_annotator import annotator_2d
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
