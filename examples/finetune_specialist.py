import os
import argparse
import torch

from util import get_dataloaders

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def dataloaders(patch_shape, batch_size, dataset, data_dir, images_dir, labels_dir):
    """Return train or val data loader for finetuning SAM.

    The data loader must be a torch data loader that retuns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reserved for background, and the IDs must be consecutive

    Here, we use `torch_em.default_segmentation_loader` for creating a suitable data loader from
    the example dataset data or from provided custom images and labels. You can either adapt this 
    for your own data (see comments below) or write a suitable torch dataloader yourself.
    """
    if dataset is not None:
        os.makedirs(data_dir, exist_ok=True)
        return get_dataloaders(patch_shape, batch_size, data_path=data_dir, dataset=dataset)

    else:
        return get_dataloaders(patch_shape, batch_size, images_dir=images_dir, labels_dir=labels_dir)

    # This will download the image and segmentation data for training if a dataset has been chosen. Otherwise,
    # the provided images and masks directories will be used to build the torch dataloader.


def run_training(checkpoint_name, model_type, dataset, data_dir, images_dir, labels_dir, save_root):
    """Run the actual model training.
    """

    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (512, 512)  # the size of patches for training
    n_iterations = 1e5  # the number of iterations to train for
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled
    device = "cuda" if torch.cuda.is_available() else "cpu"  # the device/GPU used for training. 

    # Get the dataloaders.
    train_loader, val_loader = dataloaders(patch_shape, batch_size, dataset, data_dir, images_dir, labels_dir)

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_iterations=n_iterations,
        n_objects_per_batch=n_objects_per_batch,
        device=device,
        save_root=save_root
    )
    export_model(checkpoint_name=checkpoint_name, model_type=model_type)


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    export_path = "./finetuned_specialist_model.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def finetune_specialist(args):
    """Example code for finetuning SAM on histopathology datasets."""

    model_type = args.model_type
    checkpoint_name = "sam_specialist"
    run_training(checkpoint_name=checkpoint_name, model_type=model_type, dataset=args.dataset, data_dir=args.data_dir,
                 images_dir=args.images_dir, labels_dir=args.labels_dir, save_root=args.save_root)


def main():
    """Finetune a Segment Anything model.

    This example can easily be adapted for other data (including data you have annoated with micro_sam beforehand).

    Option 1: Provide the name of a dataset implemented in torch-em for finetuning. Leave images_dir and labels_dir
    blank but provide a data path where the dataset will be downloaded to.

    Option 2: Provide images_dir and labels_dir to train from custom data. Images must be in shape (H, W, 3), labels in
    shape (H, W).

    """

    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the Histopathology datasets.")
    parser.add_argument(
        "--data_dir", default=None,
        help="(Optional) path where dataset will be loaded if dataset argument is provided.",
    )
    parser.add_argument(
        "--dataset", "-d", default=None,
        help="(Optional) Which dataset to finetune the specialist model on. Choose from datasets implemented in torch-em.",
    )
    parser.add_argument(
        "--images_dir", default=None,
        help="(Optional) Path to training images.",
    )
    parser.add_argument(
        "--labels_dir", default=None,
        help="(Optional) Path to labels corresponding to training images in images_dir.",
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="(Optional) The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h.",
    )
    parser.add_argument(
        "--save_root", "-s", default=None,
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run.",
    )
    args = parser.parse_args()
    finetune_specialist(args)


if __name__ == "__main__":
    main()
