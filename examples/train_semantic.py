import os

import torch

import torch_em
from torch_em.data import MinTwoInstanceSampler
from torch_em.data.datasets import get_pannuke_dataset

import micro_sam.training as sam_training
from micro_sam.instance_segmentation import get_unetr

from patho_sam.training import SemanticInstanceTrainer, get_train_val_split


DATA_FOLDER = "data"


def get_dataloaders(data_path):
    """This returns the PanNuke dataloaders implemented in `torch-em`.
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/histopathology/pannuke.py
    It will automatically download the PanNuke data.

    NOTE: To replace this with another data loader, you need to return a torch data loader
    that returns `x, y` tensors, where `x` is the image data and `y` are corresponding labels.
    The labels have to be in a label mask semantic segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with semantic labels for objects.
    Important: the ID 0 is reserved for background and ensure you have all semantic classes.
    """
    # All relevant stuff for the dataset.
    raw_transform = sam_training.identity  # Avoids normalizing the inputs, i.e. keeps the intensities b/w [0, 255].
    sampler = MinTwoInstanceSampler()  # Ensures that atleast one foreground class is obtained.
    label_dtype = torch.float32  # Converts labels to expected dtype.

    # Get the dataset
    dataset = get_pannuke_dataset(
        path=data_path,
        patch_shape=(1, 512, 512),
        ndim=2,
        folds=["fold_1", "fold_2"],
        custom_label_choice="semantic",
        sampler=sampler,
        label_dtype=label_dtype,
        raw_transform=raw_transform,
        download=True,
    )

    # Create custom splits.
    train_dataset, val_dataset = get_train_val_split(dataset)

    # Get the dataloaders.
    train_loader = torch_em.get_data_loader(dataset=train_dataset, batch_size=1, shuffle=True)
    val_loader = torch_em.get_data_loader(dataset=val_dataset, batch_size=1, shuffle=True)

    return train_loader, val_loader


def train_pannuke_semantic_segmentation(checkpoint_name, model_type):
    """Script for semantic segmentation for PanNuke data."""

    # Parameters for training
    num_classes = 6  # available classes are [0, 1, 2, 3, 4, 5]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_dataloaders(data_path=os.path.join(DATA_FOLDER, "pannuke"))

    # Get the trainable Segment Anything Model.
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=None,  # override to provide filepath for your trained SAM model.
    )

    # Get the UNETR model for semantic segmentation pipeline
    unetr = get_unetr(
        image_encoder=model.sam.image_encoder, device=device, out_channels=num_classes, flexible_load_checkpoint=True,
    )

    # All other stuff we need for training
    optimizer = torch.optim.AdamW(unetr.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5)

    # Converts the per-batch inputs and corresponding labels to desired format required for training.
    convert_inputs = sam_training.util.ConvertToSemanticSamInputs()

    # Trainer for semantic segmentation (implemented using 'torch_em')
    trainer = SemanticInstanceTrainer(
        name=checkpoint_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=unetr,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        log_image_interval=100,
        mixed_precision=True,
        compile_model=False,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        dice_weight=0,  # override to use weighted dice-cross entropy loss. the trainer uses cross-entropy loss only.
    )
    trainer.fit(epochs=100)


def main():
    """Finetune a Segment Anything model for semantic segmentation on the PanNuke dataset.

    This example uses image data and semantic segmentation labels for the PanNule dataset,
    but can easily be adapted for other data (including data you have annotated with patho_sam beforehand).
    NOTE: You must provide semantic class labels to train within this setup.
    """
    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use 'vit_b' here because it can be trained faster. Note that 'vit_h' yields higher quality results.
    model_type = "vit_b"

    # The name of checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'.
    checkpoint_name = "pannuke_semantic"

    train_pannuke_semantic_segmentation(checkpoint_name, model_type)


if __name__ == "__main__":
    main()
