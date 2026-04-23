import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from micro_sam.util import precompute_image_embeddings
from micro_sam.automatic_segmentation import get_predictor_and_segmenter
from patho_sam.annotation import postprocess_instance_mask, compute_object_features_parallel
from util import CELL_NAMES, ROOT


def filter_instances_by_overlap(inst_mask, bin_mask, min_pixels=1):
    inst_ids = np.unique(inst_mask)
    inst_ids = inst_ids[inst_ids != 0]

    keep = []

    bin_mask = bin_mask.astype(bool)

    for inst_id in inst_ids:
        inst_region = (inst_mask == inst_id)
        overlap = np.sum(inst_region & bin_mask)

        if overlap >= min_pixels:
            keep.append(inst_id)

    return keep


def apply_filter(inst_mask, keep_ids):
    out = np.zeros_like(inst_mask)
    for i in keep_ids:
        out[inst_mask == i] = i
    return out


def postprocess_segpath_preds(path: Path, cell_type):
    cell_type = CELL_NAMES[cell_type]
    path = Path(path) / cell_type
    volume_paths = list((path / "data").glob("*.h5"))
    predictor, _ = get_predictor_and_segmenter(model_type="vit_b_histopathology", segmentation_mode="apg")
    embedding_dir = path / "embeddings"
    df = pd.read_csv((path / f"{cell_type}_summary.csv"))
    df = df.set_index(["filename"])

    if "training_objects" not in df.keys():
        df["training_objects"] = pd.Series(pd.NA, index=df.index, dtype="Int32")

    for volume_path in tqdm(volume_paths):

        embedding_path = embedding_dir / volume_path.stem
        if not embedding_path.exists():
            continue

        if pd.notna(df.at[(volume_path.name), "training_objects"]):
            continue

        with h5py.File(volume_path, 'a') as f:
            if "labels/raw_pred" not in f:
                raise FileNotFoundError(f"Missing prediction where embedding exists in {volume_path}")

            segmentation = f["labels/raw_pred"][:]
            segmentation = postprocess_instance_mask(segmentation)
            mask_crop = f["labels/best_crop"][:]

            img = f["images/best_crop"][:]
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[:, :, :-1]
                del f["images/best_crop"]
                f.create_dataset(name="images/best_crop", data=img, compression="gzip")

            keep_ids = filter_instances_by_overlap(segmentation, mask_crop, 10)
            segmentation = apply_filter(segmentation, keep_ids)

            embeddings = precompute_image_embeddings(predictor, input_=img, save_path=embedding_path)

            seg_ids, features = compute_object_features_parallel(segmentation, embeddings)
            if len(seg_ids) != len(keep_ids):
                raise ValueError("Inconsistent number of items to be kept")

            df.loc[volume_path.name, "training_objects"] = len(seg_ids)

            f.create_dataset(name="labels/postprocessed_pred", data=segmentation, compression='gzip')
            f.create_dataset(name="object_features", data=features, compression="gzip")
            del f['labels/raw_pred']
    df.to_csv(path / f"{cell_type}_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", type=str)
    args = parser.parse_args()
    postprocess_segpath_preds(path=ROOT, cell_type=args.cell_type)


if __name__ == "__main__":
    main()
