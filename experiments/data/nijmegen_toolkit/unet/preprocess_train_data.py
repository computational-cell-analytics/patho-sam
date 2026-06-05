from pathlib import Path

import napari
import numpy as np
from natsort import natsorted
from torch_em.data.datasets.histopathology import get_ignite_dataset


def pad_to_multiple(img, mask=None, patch_size=512, mode="reflect"):
    """
    Pads image (and optional mask) so that H and W become divisible by patch_size.

    Parameters
    ----------
    img : np.ndarray
        HxWxC or HxW image.
    mask : np.ndarray or None
        HxW label mask (integer or binary).
    patch_size : int
        Target divisibility constraint (e.g. 512).
    mode : str
        Padding mode for numpy.pad. Recommended: "reflect".

    Returns
    -------
    img_pad : np.ndarray
    mask_pad : np.ndarray or None
    """

    h, w = img.shape[:2]

    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img.ndim == 3:
        img_pad = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=mode)
    else:
        img_pad = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)

    mask_pad = None
    if mask is not None:
        mask_pad = np.pad(
            mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",  # important: do NOT reflect labels
            constant_values=0,  # background
        )
    ds = get_ignite_dataset()
    return img_pad, mask_pad  # , (pad_top, pad_bottom, pad_left, pad_right)


def check_padding():
    ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/ignite/data")
    img_paths = natsorted([p for p in ROOT.glob("images/images/he/*.png") if not p.name.endswith("context.png")])
    label_paths = natsorted(
        [
            p
            for p in ROOT.glob("corrected_tissue_annotations/annotations/he/*.png")
            if not p.name.endswith("context.png")
        ]
    )
    # ds = get_ignite_dataset(
    #     "/mnt/ceph-hdd/cold/nim00020/hannibal_data/ignite", patch_shape=(128, 128), split="train", with_padding=False
    # )
    # for img, label in ds:
    #     print(img.shape)
    # breakpoint()
    import imageio.v3 as imageio
    from tqdm import tqdm

    i = 0
    for img_path in tqdm(img_paths):
        img = imageio.imread(img_path)
        width, height = img.shape[:2]
        print(width / 8, height / 8)

    print(f"{i} images would be discarded")
    return
    viewer = napari.Viewer()


check_padding()

# def crop_back(img_pad, mask_pad, pads):
#     """
#     Removes padding again.

#     Returns original-size arrays.
#     """
#     pad_top, pad_bottom, pad_left, pad_right = pads

#     h, w = img_pad.shape[:2]

#     img = img_pad[
#         pad_top:h - pad_bottom,
#         pad_left:w - pad_right
#     ]

#     if mask_pad is None:
#         return img, None

#     mask = mask_pad[
#         pad_top:h - pad_bottom,
#         pad_left:w - pad_right
#     ]

#     return img, mask
