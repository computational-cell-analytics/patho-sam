import os

import matplotlib.pyplot as plt

from torch_em.data import datasets, MinTwoInstanceSampler

from micro_sam.training import identity
from micro_sam.evaluation.model_comparison import _overlay_outline


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/data"


def for_fig_1a():
    sampler = MinTwoInstanceSampler()

    # Get a cummulative image for multiple datasets.
    get_loaders = {
        "consep": lambda: datasets.get_consep_loader(
            path=os.path.join(ROOT, "consep"), batch_size=1, shuffle=True, raw_transform=identity,
            patch_shape=(512, 512), split="test", download=True, sampler=sampler,
        ),
        # "cpm15": lambda: datasets.get_cpm_loader(
        #     path=os.path.join(ROOT, "cpm15"), batch_size=1, shuffle=True, patch_shape=(512, 512),
        #     data_choice="cpm15", resize_inputs=True, download=True, sampler=sampler, raw_transform=identity,
        # ),
        "cpm17": lambda: datasets.get_cpm_loader(
            path=os.path.join(ROOT, "cpm17"), batch_size=1, shuffle=True, patch_shape=(512, 512), split="test",
            data_choice="cpm17", resize_inputs=True, download=True, sampler=sampler, raw_transform=identity,
        ),
        "cryonuseg": lambda: datasets.get_cryonuseg_loader(
            path=os.path.join(ROOT, "cryonuseg"), batch_size=1, shuffle=True, patch_shape=(512, 512),
            split="test", resize_inputs=True, download=True, sampler=sampler, raw_transform=identity,
        ),
        "lizard": lambda: datasets.get_lizard_loader(
            path=os.path.join(ROOT, "lizard"), batch_size=1, patch_shape=(512, 512), split="test",
            resize_inputs=True, download=True, shuffle=True, sampler=sampler, raw_transform=identity,
        ),
        "lynsec": lambda: datasets.get_lynsec_loader(
            path=os.path.join(ROOT, "lynsec"), batch_size=1, patch_shape=(512, 512), shuffle=True,
            choice="h&e", resize_inputs=True, download=True, sampler=sampler, raw_transform=identity,
        ),
        "monuseg": lambda: datasets.get_monuseg_loader(
            path=os.path.join(ROOT, "monuseg"), batch_size=1, shuffle=True, patch_shape=(512, 512),
            resize_inputs=True, download=True, sampler=sampler, raw_transform=identity, split="test",
        ),
        "nuinsseg": lambda: datasets.get_nuinsseg_loader(
            path=os.path.join(ROOT, "nuinsseg"), batch_size=1, shuffle=True, patch_shape=(512, 512),
            download=True, sampler=sampler, raw_transform=identity, resize_inputs=True,
        ),
        # "pannuke": lambda: datasets.get_pannuke_loader(
        #     path=os.path.join(ROOT, "pannuke"), batch_size=1, patch_shape=(512, 512), folds=["fold_3"],
        #     download=True, shuffle=True, sampler=sampler, raw_transform=identity,
        # ),
        "puma": lambda: datasets.get_puma_loader(
            path=os.path.join(ROOT, "puma"), batch_size=1, patch_shape=(512, 512), split="test",
            download=True, sampler=sampler, raw_transform=identity, resize_inputs=True,
        ),
        "tnbc": lambda: datasets.get_tnbc_loader(
            path=os.path.join(ROOT, "tnbc"), batch_size=1, patch_shape=(512, 512), ndim=2, shuffle=True,
            split="train", resize_inputs=True, download=True, sampler=sampler, raw_transform=identity,
        )
    }

    fig, ax = plt.subplots(3, 3, figsize=(30, 30))
    ax = ax.flatten()

    for i, dname in enumerate(get_loaders.keys()):
        loader = get_loaders[dname]()
        counter = 0
        for x, y in loader:
            if counter > 0:
                break
            counter += 1

            x, y = x.squeeze().numpy(), y.squeeze().numpy()

            # Make channels last for RGB images.
            if x.shape[0] == 3:
                x = x.transpose(1, 2, 0)

            # Normalize images.
            from torch_em.transform.raw import normalize
            x = normalize(x) * 255
            x = x.astype(int)

            # Finally, plot them into one place.
            image = _overlay_outline(x, y, outline_dilation=1)

        ax[i].imshow(image, cmap="gray")
        ax[i].axis("off")

    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    plt.savefig("./fig_1a_histopathology_dataset_images.png", bbox_inches="tight")
    plt.savefig("./fig_1a_histopatholgoy_dataset_images.svg", bbox_inches="tight")


def main():
    for_fig_1a()


if __name__ == "__main__":
    main()
