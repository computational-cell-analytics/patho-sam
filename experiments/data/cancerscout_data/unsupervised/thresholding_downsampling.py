import time
from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image

# Threshold params
TARGET_MPP = 2.0
THUMB_LEVEL = None
OTSU_CHANNEL = "saturation"
MIN_TISSUE_AREA = 500
MAX_DARK_INTENSITY = 75


def open_slide(path: str):
    slide = openslide.OpenSlide(path)
    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))  # fallback: 0.25 µm/px
    print(f"[INFO] Slide: {Path(path).name}")
    print(f"[INFO] Dimensions L0: {slide.dimensions}")
    print(f"[INFO] Levels: {slide.level_count}  Downsamples: {[round(d, 1) for d in slide.level_downsamples]}")
    print(f"[INFO] MPP (L0): {mpp_x:.4f}")
    return slide, mpp_x


def choose_level(slide: openslide.OpenSlide, mpp_x: float, target_mpp: float = TARGET_MPP):
    """Selects the level whose native MPP is closest to the target MPP."""
    desired_ds = target_mpp / mpp_x
    best_level = slide.get_best_level_for_downsample(desired_ds)
    native_ds = slide.level_downsamples[best_level]
    native_mpp = mpp_x * native_ds
    print(f"[INFO] Selected level: {best_level}  (native MPP ≈ {native_mpp:.2f} µm/px, target: {target_mpp:.2f} µm/px)")
    return best_level, native_ds


def read_and_resize(
    slide: openslide.OpenSlide, level: int, native_ds: float, target_mpp: float, mpp_x: float
) -> np.ndarray:
    """
    Reads the selected level in full and optionally resizes it
    so that the output exactly matches target_mpp.
    """
    t0 = time.time()
    w, h = slide.level_dimensions[level]
    print(f"[INFO] Reading level {level} ({w}×{h} px) ...")

    region = slide.read_region((0, 0), level, (w, h))
    img = np.array(region.convert("RGB"))  # (H, W, 3) uint8

    exact_scale = (mpp_x * native_ds) / target_mpp
    if abs(exact_scale - 1.0) > 0.01:
        new_w = max(1, round(w * exact_scale))
        new_h = max(1, round(h * exact_scale))
        print(f"[INFO] Resize {w}×{h} → {new_w}×{new_h} (scale={exact_scale:.3f})")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    print(f"[OK]  Read + resize: {time.time() - t0:.1f} s  → {img.shape[1]}×{img.shape[0]} px")
    return img


def mask_red_border(img_rgb: np.ndarray, target_rgb: tuple = (254, 0, 0)) -> np.ndarray:
    """
    Returns a binary mask (255 = valid pixel, 0 = red border pixel)
    for pixels that exactly match the target RGB value.
    """
    r, g, b = target_rgb
    border_pixels = (img_rgb[:, :, 0] == r) & (img_rgb[:, :, 1] == g) & (img_rgb[:, :, 2] == b)
    border_mask = np.where(border_pixels, 0, 255).astype(np.uint8)
    n_border = border_pixels.sum()
    print(f"[INFO] Red border pixels masked out: {n_border} ({n_border / border_pixels.size:.2%} of image)")
    return border_mask


def mask_dark_bars(img_rgb: np.ndarray, max_intensity: int = MAX_DARK_INTENSITY) -> np.ndarray:
    """
    Returns a binary mask (255 = valid pixel, 0 = dark bar pixel)
    for pixels where ALL channels are below max_intensity.
    """
    dark_pixels = (
        (img_rgb[:, :, 0] <= max_intensity) & (img_rgb[:, :, 1] <= max_intensity) & (img_rgb[:, :, 2] <= max_intensity)
    )
    dark_mask = np.where(dark_pixels, 0, 255).astype(np.uint8)
    n_dark = dark_pixels.sum()
    print(f"[INFO] Dark bar pixels masked out: {n_dark} ({n_dark / dark_pixels.size:.2%} of image)")
    return dark_mask


def tissue_mask(img_rgb: np.ndarray, channel: str = OTSU_CHANNEL, min_area: int = MIN_TISSUE_AREA) -> np.ndarray:
    """
    Creates a binary tissue mask via Otsu thresholding.

    channel:
      "gray"       - classic grayscale Otsu
      "saturation" - HSV S-channel (robust against background color)
      "value"      - HSV V-channel
    """
    # ── pre-mask artifacts before Otsu sees them ──
    red_border_mask = mask_red_border(img_rgb)
    dark_bar_mask = mask_dark_bars(img_rgb)

    # ── combine: a pixel must pass both masks to be valid ──
    artifact_mask = cv2.bitwise_and(red_border_mask, dark_bar_mask)

    # ── zero out all artifact pixels so Otsu histogram is not skewed ──
    img_clean = img_rgb.copy()
    img_clean[artifact_mask == 0] = 255  # set to white background

    # ── Otsu on cleaned image ──
    if channel == "gray":
        ch = cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elif channel in ("saturation", "value"):
        hsv = cv2.cvtColor(img_clean, cv2.COLOR_RGB2HSV)
        ch = hsv[:, :, 1] if channel == "saturation" else hsv[:, :, 2]
        _, mask = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:
        raise ValueError(f"Unknown channel: {channel}")

    # ── apply border mask on top of Otsu result ──
    mask = cv2.bitwise_and(mask, red_border_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    coverage = clean.sum() / 255 / clean.size * 100
    print(f"[OK]  Tissue mask: {coverage:.1f} % tissue coverage  (channel: {channel})")
    return clean


def save_results(img_rgb: np.ndarray, mask: np.ndarray, out_dir: Path, alpha: float = 0.40):

    Image.fromarray(img_rgb).save(out_dir / "thumbnail.png")

    Image.fromarray(mask).save(out_dir / "tissue_mask.png")

    print(f"[OK]  Saved to: {out_dir.resolve()}")


def run_pipeline(wsi_path: str, output_path, target_mpp):
    print("\n" + "=" * 55)
    print("  WSI TISSUE-DETECTION PIPELINE")
    print("=" * 55)

    slide, mpp_x = open_slide(wsi_path)
    level, native_ds = choose_level(slide, mpp_x, target_mpp)
    img = read_and_resize(slide, level, native_ds, target_mpp, mpp_x)

    mask = tissue_mask(img, channel=OTSU_CHANNEL, min_area=MIN_TISSUE_AREA)
    save_results(img, mask, output_path)

    slide.close()
    print("=" * 55 + "\n")
    return img, mask
