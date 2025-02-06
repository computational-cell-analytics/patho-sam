import imageio.v3 as imageio

from micro_sam.sam_annotator import annotator_2d


def run_annotator_with_finetuned_model():
    """Run the 2d annotator with a custom (finetuned) model.

    Here, we use the model that is produced by `finetune_nuclick.py` and apply it
    for an image from the validation set.
    """
    # take the last frame, which is part of the val set, so the model was not directly trained on it
    im = imageio.imread("./data/IHC_nuclick/IHC/images/Validation/ROI_338_2.png")

    # set the checkpoint and the path for caching the embeddings
    checkpoint = "./finetuned_nuclick_model.pth"
    embedding_path = "./embeddings/embeddings-finetuned.zarr"

    # Adapt this if you finetune a different model type, e.g. vit_h.
    model_type = "vit_b"  # We finetune a vit_b in the example script.

    # Run the 2d annotator with the custom model.
    annotator_2d(im, model_type=model_type, embedding_path=embedding_path, checkpoint=checkpoint)


if __name__ == "__main__":
    run_annotator_with_finetuned_model()
