import argparse
import json
from pathlib import Path
from shutil import rmtree
from typing import Tuple

import h5py
import numpy as np
import pyvips
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

from patho_sam.annotation import postprocess_instance_mask

SQUARE_LENGTH = 2048


ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data/"


def process_selected_rois(
    pyramid_path: Path,
    h5_path: Path,
    predictor,
    segmenter,
    embedding_dir: Path,
    roi_position: Tuple[int, int] = None,
    roi_height: int = SQUARE_LENGTH,
    roi_width: int = SQUARE_LENGTH,
    dry: bool = False,
):
    tile_shape, halo = (384, 384), (64, 64)

    with h5py.File(h5_path, "a") as f:
        if "inst_pred" in f:
            print("Instance prediction already computed.")
            return

        if "img" in f.keys():
            img = f["img"][:]
            img = img.transpose(1, 2, 0)
            print("image crop already extracted")
            if dry:
                return
        else:
            print(f"Extracting crop for {h5_path.stem}")
            image = pyvips.Image.new_from_file(pyramid_path, access="sequential")
            patch = image.crop(roi_position[0], roi_position[1], roi_width, roi_height)
            img = np.ndarray(
                buffer=patch.write_to_memory(), dtype=np.uint8, shape=[patch.height, patch.width, patch.bands]
            )
            f.create_dataset(name="img", data=img.transpose(2, 0, 1), compression="gzip")
            if dry:
                return
        if embedding_dir.exists():
            rmtree(embedding_dir)

        inst_pred = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            embedding_path=embedding_dir,
            halo=halo,
            tile_shape=tile_shape,
            ndim=2,
            batch_size=4,  # Apparently 16 is too much for apg?
            verbose=True,
            optimize_memory=True,
            input_path=img,
        )
        inst_pred = postprocess_instance_mask(inst_pred)

        f.create_dataset(name="inst_pred", data=inst_pred, compression="gzip")


def get_roi_preds(data_path: Path, split, entity):
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b_histopathology", segmentation_mode="apg", is_tiled=True
    )

    image_dir = data_path / f"{split}_models" / "CancerScout_Lung" / entity
    json_path = data_path / f"{split}_models" / f"{split}_rois.json"
    h5_dir = data_path / f"{split}_models" / f"{entity}_data" / "fixed_h5_files"
    h5_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = h5_dir.parent / "embeddings"
    embedding_dir.mkdir(exist_ok=True, parents=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    for file_name, roi in data[entity].items():
        if not roi:
            print("No roi provided for: ", file_name)
            continue
        file_name = file_name.replace("_pyramid.tiff", "")
        img_path = image_dir / (file_name + ".tiff")
        if not img_path.exists():
            raise FileNotFoundError
        process_selected_rois(
            img_path,
            h5_dir / (file_name + ".h5"),
            predictor=predictor,
            segmenter=segmenter,
            embedding_dir=embedding_dir / file_name,
            roi_position=roi,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--entity", type=str, default="new_tumor")

    args = parser.parse_args()

    get_roi_preds(data_path=Path(args.data_path), split=args.split, entity=args.entity)


if __name__ == "__main__":
    main()
