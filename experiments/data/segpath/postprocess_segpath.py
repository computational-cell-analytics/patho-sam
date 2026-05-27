import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from shutil import rmtree

import h5py
import numpy as np
import pandas as pd
from micro_sam.automatic_segmentation import get_predictor_and_segmenter
from micro_sam.util import precompute_image_embeddings
from tqdm import tqdm
from util import CELL_NAMES, ROOT

from patho_sam.annotation import compute_object_features_parallel, postprocess_instance_mask


def filter_instances_by_overlap(inst_mask, bin_mask, min_pixels=1):
    inst_ids = np.unique(inst_mask)
    inst_ids = inst_ids[inst_ids != 0]

    keep = []

    bin_mask = bin_mask.astype(bool)

    for inst_id in inst_ids:
        inst_region = inst_mask == inst_id
        overlap = np.sum(inst_region & bin_mask)

        if overlap >= min_pixels:
            keep.append(inst_id)

    return keep


def apply_filter(inst_mask, keep_ids):
    out = np.zeros_like(inst_mask)
    for i in keep_ids:
        out[inst_mask == i] = i
    return out


def postprocess_sample(paths, df, predictor, overwrite):
    embedding_path = paths[1]
    file_path = paths[0]

    with h5py.File(file_path, "a") as f:
        if "object_features" in f and not overwrite:
            obj_feat = f["object_features"][:]
            if embedding_path.exists():
                rmtree(embedding_path)
            return file_path.name, obj_feat.shape[0]

        if "labels/raw_pred" not in f:
            if embedding_path.exists():
                raise FileNotFoundError(f"Missing prediction where embedding exists in {file_path}")

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

        seg_ids, features = compute_object_features_parallel(
            segmentation=segmentation, image_embeddings=embeddings, verbose=False, n_workers=max(1, cpu_count() // 4)
        )

        if len(seg_ids) != features.shape[0]:
            raise ValueError("Inconsistent number of seg ids to features")

        f.create_dataset(name="labels/postprocessed_pred", data=segmentation, compression="gzip")
        f.create_dataset(name="object_features", data=features, compression="gzip")
        del f["labels/raw_pred"]
        rmtree(embedding_path)

        return file_path.name, len(seg_ids)


def postprocess_segpath_preds(path: Path, cell_type: str, overwrite: bool = False):
    cell_type = CELL_NAMES[cell_type]
    path = Path(path) / cell_type
    volume_paths = list((path / "data").glob("*.h5"))
    predictor, _ = get_predictor_and_segmenter(model_type="vit_b_histopathology", segmentation_mode="apg")
    embedding_dir = path / "embeddings"
    df = pd.read_csv((path / f"{cell_type}_summary.csv"))
    df = df.set_index(["filename"])
    df = df[df["predicted"]]
    if overwrite or "training_objects" not in df.columns:
        df["training_objects"] = pd.Series(pd.NA, index=df.index, dtype="Int32")

    filtered_indices = df[df["training_objects"].isna()].index.tolist()

    volume_paths = [(p, embedding_dir / p.stem) for p in volume_paths if p.name in filtered_indices]

    _postprocess_sample = partial(postprocess_sample, df=df, predictor=predictor, overwrite=overwrite)

    with Pool(4) as p:
        print("using cpu cores: ", (cpu_count() - 2))
        results = np.array(
            list(
                tqdm(
                    p.imap(_postprocess_sample, volume_paths),
                    total=len(volume_paths),
                    desc=f"Postprocessing {cell_type}",
                )
            )
        )

    mapping = dict(results)

    df["training_objects"] = df.index.map(mapping)

    df.to_csv(path / f"{cell_type}_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=ROOT)
    parser.add_argument("--cell_type", type=str, default=None)
    args = parser.parse_args()

    if args.cell_type is None:
        for cell_type in CELL_NAMES.keys():
            postprocess_segpath_preds(path=args.input_path, cell_type=cell_type)
    else:
        postprocess_segpath_preds(path=args.input_path, cell_type=args.cell_type)


if __name__ == "__main__":
    main()
