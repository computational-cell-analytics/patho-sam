import os
from typing import Tuple
import pyvips
import argparse
import torch
import numpy as np
import h5py
from glob import glob
from collections import OrderedDict
from micro_sam.util import precompute_image_embeddings, get_sam_model, get_device
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/"

SQUARE_LENGTH = 2048

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


def get_segmentation(predictor, segmenter, img, tile_shape, halo, embedding_path=None):

    return automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        embedding_path=embedding_path,
        halo=halo,
        tile_shape=tile_shape,
        ndim=2,
        batch_size=16,
        verbose=True,
        optimize_memory=True,
        input_path=img
    )


def process_selected_rois(predictor, segmenter, embedding_dir, img_path, output_dir, roi_position: Tuple[int, int] = None,
                          roi_height=SQUARE_LENGTH, roi_width=SQUARE_LENGTH):
    tile_shape, halo = (384, 384), (64, 64)
    os.makedirs(output_dir, exist_ok=True)

    h5_path = os.path.join(output_dir, os.path.basename(img_path.replace(" ", "").split(".")[0])) + ".h5"

    with h5py.File(h5_path, 'a') as f:
        if "inst_pred" in f.keys():
            print("Instance prediction already computed.")
            return

        if "img" in f.keys():
            img = f["img"][:]
            return
        else:
            image = pyvips.Image.new_from_file(img_path, access='sequential')
            patch = image.crop(roi_position[0], roi_position[1], roi_width, roi_height)
            img = np.ndarray(buffer=patch.write_to_memory(),
                             dtype=np.uint8,
                             shape=[patch.height, patch.width, patch.bands])
            f.create_dataset(name="img", data=img, compression="gzip")
            return

        if os.path.exists(embedding_dir):
            if len(os.listdir(embedding_dir)) > 0:
                inst_pred = get_segmentation(predictor, segmenter, img_path, tile_shape, halo,
                                             embedding_path=embedding_dir)
        else:
            compute_embeddings_for_roi(predictor=predictor,
                                       roi=img,
                                       tile_shape=tile_shape,
                                       halo=halo,
                                       roi_embeddings_dir=embedding_dir
                                       )

            inst_pred = get_segmentation(predictor, segmenter, img, tile_shape, halo, embedding_path=embedding_dir)
        f.create_dataset(name="inst_pred", data=inst_pred, compression="gzip")


def get_rois(img_path, embedding_dir, checkpoint_path, output_dir, rois):
    predictor, segmenter = get_instance_segmentation_model(checkpoint_path)
    if os.path.isdir(img_path) and isinstance(rois, list):
        for path, roi in zip(glob(os.path.join(img_path, "*"))), rois:
            process_selected_rois(predictor, segmenter, embedding_dir, path, output_dir, roi)
    elif os.path.isfile(img_path) and isinstance(rois, tuple):
        process_selected_rois(predictor, segmenter, embedding_dir, img_path, output_dir, rois)
    else:
        raise ValueError("If img_path is a directory, rois must be a list of coordinate tuples (x1, y1)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default=ROOT)
    parser.add_argument("--embedding_dir", type=str)
    parser.add_argument("--checkpoint_path", "-c", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    get_rois(
        img_dir=args.img_path,
        embedding_dir=args.embedding_dir,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        rois=
        )


if __name__ == "__main__":
    main()
