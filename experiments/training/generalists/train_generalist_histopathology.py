import os
import argparse

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from get_generalist_datasets import get_generalist_hp_loaders


def finetune_generalist(args):
    """Example code for finetuning SAM on histopathology datasets."""
    # override this (below) if you have some more complex set-up and need to specify the exact GPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint.
    patch_shape = (512, 512)  # the patch shape for training.
    n_objects_per_batch = args.n_objects  # the number of objects per batch that will be sampled.
    checkpoint_name = f"{args.model_type}/patho_sam"

    # all the stuff we need for training
    train_loader, val_loader = get_generalist_hp_loaders(patch_shape=patch_shape, data_path=args.input_path)
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10, "verbose": True}

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=None,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        with_segmentation_decoder=True,
        device=device,
        lr=1e-5,
        n_iterations=args.iterations,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        verify_n_labels_in_loader=None,  # NOTE: Setting to 'None' verifies all labels in the loader(s).
    )

    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the Histopathology datasets.")
    parser.add_argument(
        "--input_path", "-i", default="/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/data",
        help="The filepath to the datasets. If the data does not exist yet it will be downloaded.",
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h.",
    )
    parser.add_argument(
        "--save_root", "-s", default=None,
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run.",
    )
    parser.add_argument(
        "--iterations", type=int, default=int(25e4),
        help="For how many iterations should the model be trained? By default 250k.",
    )
    parser.add_argument(
        "--export_path", "-e",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools.",
    )
    parser.add_argument(
        "--n_objects", type=int, default=25, help="The number of instances (objects) per batch used for finetuning."
    )
    args = parser.parse_args()
    finetune_generalist(args)


if __name__ == "__main__":
    main()
