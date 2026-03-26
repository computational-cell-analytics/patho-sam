import os
import pyvips
from tqdm import tqdm
import argparse
import torch
import imageio.v3 as imageio
import numpy as np
import zarr
import json
from collections import OrderedDict
from micro_sam.util import precompute_image_embeddings, get_sam_model, get_device
from pathlib import Path
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/"

SQUARE_LENGTH = 5120


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


def compute_embeddings_for_roi(predictor, roi, tile_shape, halo, roi_embeddings_dir):
    return precompute_image_embeddings(predictor=predictor,
                                       input_=roi,
                                       tile_shape=tile_shape,
                                       halo=halo,
                                       save_path=roi_embeddings_dir,
                                       verbose=True,
                                       batch_size=12,
                                       ndim=2,
                                       )


def get_segmentation(predictor, segmenter, img_path, tile_shape, halo, output_path, embedding_path=None):

    automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        embedding_path=embedding_path,
        halo=halo,
        tile_shape=tile_shape,
        ndim=2,
        batch_size=16,
        verbose=True,
        optimize_memory=True,
        output_path=output_path,
        input_path=img_path
    )


def process_selected_rois(data_root, img_type: str, coordinate_dict: dict, predictor, segmenter, split, ckp):
    tile_shape, halo = (384, 384), (64, 64)
    rois_dir = os.path.join(data_root, f"{split}_models", f"rois_{img_type}")
    roi_images_dir = os.path.join(rois_dir, "images")
    rois_embeddings_dir = os.path.join(rois_dir, "embeddings")
    segmentation_dir = os.path.join(rois_dir, f"segmentations_{Path(ckp).parts[-3]}")
    print(segmentation_dir)
    os.makedirs(rois_embeddings_dir, exist_ok=True)
    os.makedirs(roi_images_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    for img_name, roi_position in tqdm(coordinate_dict.items()):
        non_pyramid_path = os.path.join(data_root, f"{split}_models", "CancerScout_Lung", img_type,
                                        img_name.replace("_pyramid.tiff", ".tiff"))
        roi_name = os.path.basename(non_pyramid_path.split(".")[0])
        pred_outpath = os.path.join(segmentation_dir, f"{roi_name}.tiff")
        if os.path.exists(pred_outpath):
            continue
        embedding_dir = os.path.join(rois_embeddings_dir, roi_name)
        img_path = os.path.join(roi_images_dir, f"roi_{roi_name}.tiff")

        if not os.path.exists(img_path):
            image = pyvips.Image.new_from_file(non_pyramid_path, access='sequential')
            patch = image.crop(roi_position[0], roi_position[1], SQUARE_LENGTH, SQUARE_LENGTH)
            patch_np = np.ndarray(buffer=patch.write_to_memory(),
                                  dtype=np.uint8,
                                  shape=[patch.height, patch.width, patch.bands])
            imageio.imwrite(img_path, patch_np,  plugin="tifffile", compression="zlib")
        else:
            patch_np = imageio.imread(img_path)

        if os.path.exists(embedding_dir):
            if len(os.listdir(embedding_dir)) > 0:
                get_segmentation(predictor, segmenter, img_path, tile_shape, halo, output_path=pred_outpath,
                                 embedding_path=embedding_dir)
                continue

        compute_embeddings_for_roi(predictor=predictor,
                                   roi=patch_np,
                                   tile_shape=tile_shape,
                                   halo=halo,
                                   roi_embeddings_dir=embedding_dir
                                   )

        get_segmentation(predictor, segmenter, img_path, tile_shape, halo, output_path=pred_outpath,
                         embedding_path=embedding_dir)


def get_rois(data_root, img_type, split, checkpoint_path, json_root):
    predictor, segmenter = get_instance_segmentation_model(checkpoint_path)
    with open(os.path.join(json_root, f"{split}_rois.json"), 'r') as f:
        roi_dict = json.load(f)
    process_selected_rois(data_root, img_type, roi_dict[img_type], predictor, segmenter, split, checkpoint_path)
    print(f"Finished processing {img_type} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=ROOT)
    parser.add_argument("--img_type", type=str)
    parser.add_argument("--split", choices=["train", "eval"])
    parser.add_argument("--checkpoint_path", "-c", type=str, default=None)
    parser.add_argument("--json_root", type=str,
                        default="/user/titus.griebel/u23324/patho-sam/experiments/data/cancerscout_data")
    args = parser.parse_args()

    get_rois(
        data_root=args.data_root,
        img_type=args.img_type,
        split=args.split,
        checkpoint_path=args.checkpoint_path,
        json_root=args.json_root,
    )


if __name__ == "__main__":
    main()
