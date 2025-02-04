import os

import torch

import torch_em
from torch_em.data import MinTwoInstanceSampler
from torch_em.data.datasets import get_pannuke_dataset
from util import _get_train_val_split
import micro_sam.training as sam_training
from micro_sam.instance_segmentation import get_unetr

from patho_sam.training import SemanticInstanceTrainer


def get_dataloaders(patch_shape, data_path):
    """This returns the PanNuke data loaders implemented in `torch-em`.
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/histopathology/pannuke.py
    It will automatically download the PanNuke data.

    NOTE: To replace this with another data loader, you need to return a torch data loader
    that returns `x, y` tensors, where `x` is the image data and `y` are corresponding labels.
    The labels have to be in a label mask semantic segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with semantic labels for objects.
    Important: the ID 0 is reserved for background and ensure you have all semantic classes.
    """
    raw_transform = sam_training.identity
    sampler = MinTwoInstanceSampler()
    label_dtype = torch.float32

    dataset = get_pannuke_dataset(
        path=data_path,
        patch_shape=patch_shape,
        ndim=2,
        folds=["fold_1", "fold_2"],
        custom_label_choice="semantic",
        sampler=sampler,
        label_dtype=label_dtype,
        raw_transform=raw_transform,
        download=True,
    )

    # Create custom splits.
    train_dataset, val_dataset = _get_train_val_split(dataset)

    # Get the dataloaders.
    train_loader = torch_em.get_data_loader(train_dataset, batch_size=8, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(val_dataset, batch_size=1, shuffle=True, num_workers=16)

    return train_loader, val_loader


def train_pannuke_semantic_segmentation(args):
    """Code for semantic instance segmentation for PanNuke data.
    """
    # Hyperparameters for training
    model_type = args.model_type
    num_classes = 6  # available classes are [0, 1, 2, 3, 4, 5]
    checkpoint_path = args.checkpoint_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_name = f"{model_type}/pannuke_semantic"

    train_loader, val_loader = get_dataloaders(
        patch_shape=(1, 512, 512), data_path=os.path.join(args.input_path, "pannuke")
    )
    #  This freezes the image encoder, prompt encoder and mask decoder so only the semantic decoder is finetuned
    freeze = ["image_encoder", "prompt_encoder", "mask_decoder"]
    checkpoint_name += "/finetune_decoder_only_from_scratch"

    # Get the trainable Segment Anything Model.
    model, state = sam_training.get_trainable_sam_model(
        model_type=model_type, device=device, checkpoint_path=checkpoint_path, freeze=freeze, return_state=True,
    )

    #  This forces decoder finetuning from scratch, i. e. without pre-trained weights
    decoder_state = None

    # Get the UNETR model for semantic segmentation pipeline
    unetr = get_unetr(
        image_encoder=model.sam.image_encoder,
        decoder_state=decoder_state,
        device=device,
        out_channels=num_classes,
        flexible_load_checkpoint=True,
    )

    # All other stuff we need for training
    optimizer = torch.optim.AdamW(unetr.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5)

    # This class creates all the training data for each batch (inputs and semantic labels)
    convert_inputs = sam_training.util.ConvertToSemanticSamInputs()

    # The trainer which performs the semantic segmentation training and validation (implemented using 'torch_em')
    trainer = SemanticInstanceTrainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=unetr,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        log_image_interval=10,
        mixed_precision=True,
        compile_model=False,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        dice_weight=0,
    )
    trainer.fit(iterations=int(args.iterations), overwrite_training=False)


def main(args):
    """Finetune a Segment Anything model for semantic segmentation on the PanNuke dataset.

    """
    train_pannuke_semantic_segmentation(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", default=None, type=str,
        help="Path where you would like to store the PanNuke data."
    )
    parser.add_argument(
        "-m", "--model_type", default="vit_b_histopathology", type=str,
        help="(Optional) The choice of model to perform semantic segmentation finetuning on."
    )
    parser.add_argument(
        "-c", "--checkpoint_path", default=None, type=str,
        help="(Optional) Filepath to the model with which you would like to perform downstream semantic segmentation."
    )
    parser.add_argument(
        "-s", "--save_root", default=None, type=str,
        help="Filepath where to store the trained model checkpoints and logs."
    )
    parser.add_argument(
        "--iterations", default=1e5, type=str,
        help="The total number of iterations to train the model for."
    )
    args = parser.parse_args()
    main(args)
