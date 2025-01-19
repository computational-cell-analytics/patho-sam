import os
from tqdm import tqdm

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_pannuke_loader

from micro_sam.util import get_sam_model
import micro_sam.training as sam_training
from micro_sam.instance_segmentation import get_unetr


def evaluate_pannuke_semantic_segmentation(args):
    # Stuff needed for inference
    model_type = args.model_type
    num_classes = 6  # available classes are [0, 1, 2, 3, 4, 5]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the dataloaders
    raw_transform = sam_training.identity
    sampler = MinInstanceSampler()
    label_dtype = torch.float32

    loader = get_pannuke_loader(
        path=os.path.join(args.input_path, "pannuke"),
        batch_size=1,
        patch_shape=(1, 256, 256),
        ndim=2,
        folds=["fold_1"],
        custom_label_choice="semantic",
        sampler=sampler,
        label_dtype=label_dtype,
        raw_transform=raw_transform,
        download=True,
    )

    # Get the SAM model
    predictor = get_sam_model(model_type=model_type, device=device)

    # Get the UNETR model for semantic segmentation pipeline
    state = torch.load(args.checkpoint_path, map_location="cpu")
    decoder_state = {k: v for k, v in state["model_state"].items() if not k.startswith("encoder")}
    unetr = get_unetr(
        image_encoder=predictor.model.image_encoder,
        out_channels=num_classes,
        decoder_state=decoder_state,
        device=device,
    )

    for x, y in tqdm(loader):

        # Run inference
        outputs = unetr(x.to(device))

        # Perform argmax to get the outputs
        masks = torch.argmax(outputs, dim=1)
        masks = masks.detach().cpu().numpy().squeeze()

        # Plot images
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(30, 15))

        image = x.squeeze().numpy().transpose(1, 2, 0)
        image = image.astype(int)
        ax[0].imshow(image)
        ax[0].axis("off")

        gt = y.squeeze().numpy()
        ax[1].imshow(gt)
        ax[1].axis("off")

        ax[2].imshow(masks)
        ax[2].axis("off")

        plt.savefig("./test.png")
        plt.close()

        breakpoint()


def main(args):
    evaluate_pannuke_semantic_segmentation(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/test/data", type=str)
    parser.add_argument("-m", "--model_type", default="vit_b", type=str)
    parser.add_argument("-c", "--checkpoint_path", default=None, type=str)
    args = parser.parse_args()
    main(args)
