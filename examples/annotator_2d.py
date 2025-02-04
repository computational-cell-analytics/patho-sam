import os

from tukra.io import read_image

from micro_sam.sam_annotator import annotator_2d


# DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def whole_slide_image_annotator(use_finetuned_model):
    """Run the 2d annotator for an example image from the UWaterloo Skin dataset.

    See https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection for details on the data.
    """
    # example_data = fetch_dermoscopy_example_data(DATA_CACHE)

    fpath = "./CMU-1.svs"

    image = read_image(fpath, image_size=(10000, 10000, 10000, 10000), scale=None)

    print(image.shape)

    import napari
    v = napari.Viewer()
    v.add_image(image)
    napari.run()

    breakpoint()

    if use_finetuned_model:
        model_type = "vit_b_histopathology"
    else:
        model_type = "vit_b"

    annotator_2d(image=image, model_type=model_type)


def main():
    # Whether to use the fine-tuned SAM model for WSIs.
    use_finetuned_model = True

    # 2d annotator for uwaterloo-skin data
    whole_slide_image_annotator(use_finetuned_model)


if __name__ == "__main__":
    main()
