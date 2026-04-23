import os
import numpy as np
import json
import imageio.v3 as imageio
from tqdm import tqdm
import torch
from collections import OrderedDict
from natsort import natsorted
from skimage.measure import regionprops
from skimage.measure import label as connected_components
from micro_sam.automatic_segmentation import get_predictor_and_segmenter
from micro_sam.util import get_sam_model, precompute_image_embeddings, get_device
from micro_sam.prompt_based_segmentation import segment_from_points

NIJMEGEN_ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/ignite_nijmegen/"


def get_instance_segmentation_model(checkpoint_path, device=get_device()):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state"]
    if any(k.startswith("sam.image_encoder") for k in state.keys()):
        predictor, segmenter = get_predictor_and_segmenter(
            model_type="vit_b_histopathology", is_tiled=True, checkpoint=checkpoint_path,
        )
    else:
        predictor = get_sam_model(model_type="vit_b_histopathology", device=device)
        encoder_state = OrderedDict(
            [(k[len("encoder."):], v) for k, v in state.items() if k.startswith("encoder")]
        )
        predictor.model.image_encoder.load_state_dict(encoder_state)

        decoder_state = {"decoder_state": OrderedDict(
            [(k, v) for k, v in state.items() if not k.startswith("encoder")]
        )}
        predictor, segmenter = get_predictor_and_segmenter(
            model_type="vit_b_histopathology", is_tiled=True,
            predictor=predictor, state=decoder_state, device=device,
            segmentation_mode="ais",
        )
    return predictor, segmenter


def predict_ignite_with_pathosam(type: str, model_path):
    with open(os.path.join(NIJMEGEN_ROOT, f"modified_ignite_{type}.json"), 'r') as f:
        selection_dict = json.load(f)

    predictor, _ = get_instance_segmentation_model(
        checkpoint_path=model_path,
    )

    pred_output_dir = os.path.join(NIJMEGEN_ROOT, "predictions")
    img_output_dir = os.path.join(NIJMEGEN_ROOT, "annotated_images")
    embedding_dir = os.path.join(NIJMEGEN_ROOT, "embeddings")
    os.makedirs(pred_output_dir, exist_ok=True)
    os.makedirs(img_output_dir, exist_ok=True)

    selected_images = natsorted([key for key in selection_dict.keys()
                                 if selection_dict[key]["centroids_xy"] is not None])

    for img_name in tqdm(selected_images):
        img_data = selection_dict[img_name]
        centroid_prompts = img_data["centroids_xy"]

        pred_path = os.path.join(pred_output_dir, img_name.replace(".png", "_pred.tiff"))  # For predicted samples, skip
        if os.path.exists(pred_path):
            continue

        img = imageio.imread(os.path.join(NIJMEGEN_ROOT, "images", "pdl1", type, img_name))

        x1, y1, x2, y2 = img_data["annotated_xyxy"]
        img = img[y1:y2, x1:x2][:, :, :3]  # Crop to annotated region and remove alpha channel

        embedding_path = os.path.join(embedding_dir, img_name.split(".")[0])

        img_embeddings = precompute_image_embeddings(
            predictor=predictor,
            input_=img,
            ndim=2,
            verbose=False,
            save_path=embedding_path,
            tile_shape=(384, 384),
            halo=(64, 64)
        )

        masks = [
            segment_from_points(
                predictor=predictor,
                points=np.expand_dims(np.array([round(centroid_coords[1]), round(centroid_coords[0])]), axis=0),
                labels=np.array([1]),
                image_embeddings=img_embeddings
            ) for centroid_coords in tqdm(centroid_prompts)
        ]

        # Merge all segmentations into one.

        # 1. First, we get the area per object and try to map as: big objects first and small ones then
        #    (to avoid losing tiny objects near-by or to overlaps)
        mask_props = []

        for mask in masks:
            if not mask.any():
                continue

            coords = np.argwhere(mask)
            minr, minc = coords.min(axis=0)
            maxr, maxc = coords.max(axis=0) + 1  # slices are exclusive at the end

            # 2. Crop to bounding box
            submask = mask[minr:maxr, minc:maxc]

            # 3. Connected components
            labeled_cc, _ = connected_components(submask)
            props = regionprops(labeled_cc)
            if not props:
                continue

            area = max(p.area for p in props)

            mask_props.append({
                "mask": submask,
                "area": area,
                "bbox": (minr, minc, maxr, maxc)
            })
        assorted_masks = sorted(mask_props, key=lambda x: x["area"], reverse=True)
        masks = [m["mask"] for m in assorted_masks]

        segmentation = np.zeros(img.shape[:2], dtype=np.uint32)

        for j, per_mask in enumerate(assorted_masks, start=1):
            minr, minc, maxr, maxc = per_mask["bbox"]
            submask = per_mask["mask"]
            subseg = segmentation[minr:maxr, minc:maxc]
            subseg[submask > 0] = j

        imageio.imwrite(pred_path, segmentation,
                        compression="zlib")
        imageio.imwrite(os.path.join(img_output_dir, img_name.replace(".png", ".tiff")), img)


def main():
    for type in ["pdl1", "nuclei"]:
        predict_ignite_with_pathosam(
            type,
            "/mnt/ceph-hdd/cold/nim00020/hannibal_data/pathosam-models/v3/instance_segmentation/best.pt")


if __name__ == "__main__":
    main()
