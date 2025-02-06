import os

import torch

import torch_em
from torch_em.data import MinTwoInstanceSampler
from torch_em.data.datasets import get_nuclick_dataset
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from patho_sam.training import get_train_val_split, histopathology_identity


DATA_FOLDER = "data"


def get_dataloaders(batch_size, patch_shape, train_instance_segmentation):
    """This returns the NuClick dataloaders implemented in `torch-em`.
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/histopathology/nuclick.py
    It will automatically download the NuClick data.

    NOTE: To replace this with another data loader, you need to return a torch data loader
    that returns `x, y` tensors, where `x` is the image data and `y` are corresponding labels.
    The labels have to be in a label mask semantic segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with semantic labels for objects.
    Important: the ID 0 is reserved for background, and the IDS must be consecutive.

    See https://github.com/computational-cell-analytics/micro-sam/blob/master/examples/finetuning/finetune_hela.py
    for more details on how to create your custom dataloaders.
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # All relevant stuff for the dataset.
    raw_transform = histopathology_identity  # Avoids normalizing the inputs, i.e. keeps the intensities b/w [0, 255].
    sampler = MinTwoInstanceSampler()  # Ensures that atleast one foreground class is obtained.
    label_dtype = torch.float32  # Converts labels to expected dtype.

    if train_instance_segmentation:
        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
        )
    else:
        label_transform = torch_em.transform.label.connected_components

    dataset = get_nuclick_dataset(
        path=DATA_FOLDER,
        patch_shape=patch_shape,
        split="Train",
        sampler=sampler,
        label_dtype=label_dtype,
        label_transform=label_transform,
        raw_transform=raw_transform,
        download=True,  # This will download the image and segmentation data for training.
    )

    # Get the datasets.
    train_ds, val_ds = get_train_val_split(dataset)

    # Get the dataloaders.
    train_loader = torch_em.get_data_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def run_training(checkpoint_name, model_type, train_instance_segmentation):
    """Run the actual model training."""

    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (512, 512)  # the size of patches for training
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled
    device = "cuda" if torch.cuda.is_available() else "cpu"  # the device/GPU used for training.

    # Get the dataloaders.
    train_loader, val_loader = get_dataloaders(batch_size, patch_shape, train_instance_segmentation)

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
    )


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    export_path = "./finetuned_nuclick_model.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(checkpoint_path=checkpoint_path, model_type=model_type, save_path=export_path)


def main():
    """Finetune a Segment Anything model.

    This example uses image data and segmentations from the NuClick dataset for lymphocyte segmentation,
    but can be easily adapted for other data (including data you have annotated with 'micro_sam' beforehand).
    """
    # The 'model_type' determines which base model is used to initialize the weights that are finetuned.
    # We use 'vit_b' here becaise it can be trained faster. Note that 'vit_h' usually yields higher quality results.
    model_type = "vit_b"

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "sam_nuclick"

    # Train an additional convolutional decoer for end-to-end automatic instance segmentation.
    train_instance_segmentation = True

    run_training(checkpoint_name, model_type, train_instance_segmentation)
    export_model(checkpoint_name, model_type)


if __name__ == "__main__":
    main()
