from pathlib import Path
from typing import List
import h5py
import argparse
from tqdm import tqdm
import pandas as pd

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from util import ROOT, CELL_NAMES


def get_instance_predictions_and_embeddings_segpath(path=ROOT, cell_type: str = None, n_images=None,
                                                    organs: List = None, maximize_instances: bool = True,
                                                    min_instances: int = 0):
    cell_type = CELL_NAMES[cell_type]
    data_path = Path(path) / cell_type / "data"
    embedding_dir = Path(path) / cell_type / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    df = pd.read_csv(Path(ROOT) / cell_type / f"{cell_type}_summary.csv", index_col="filename")

    # We only predict for images where binary masks exist
    filtered_df = df[df["num_instances"] > min_instances]

    # We check for already predicted images and substract them from n_images
    if "training_objects" in df.keys():
        obj_feat = df["training_objects"]
        n_images -= (obj_feat > 0).sum()
        n_images = max(0, n_images)
        filtered_df = filtered_df[(filtered_df["training_objects"] < 1) | (filtered_df["training_objects"].isna())]

    if organs is not None:
        filtered_df = df[df["tumour_type"].isin(organs)]

    if maximize_instances:
        filtered_df = filtered_df.nlargest(n_images, "num_instances")

    else:
        filtered_df = filtered_df.sample(n_images, random_state=42)

    predict_samples = filtered_df.index.tolist()

    volume_paths = [data_path / predict_sample for predict_sample in predict_samples]

    predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_histopathology", segmentation_mode="apg")

    for volume_path in tqdm(volume_paths):
        embedding_path = embedding_dir / volume_path.stem

        if embedding_path.exists():
            continue

        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=volume_path, key="images/best_crop",
            embedding_path=embedding_path,
            ndim=2, verbose=False)

        with h5py.File(volume_path, 'a') as f:
            if "labels/pathosam_instance_pred_crop" in f:
                continue
            f.create_dataset(name="labels/raw_pred", data=segmentation, compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", type=str, default=None)
    parser.add_argument("--n_images", type=int, default=100)
    args = parser.parse_args()
    for cell_type in CELL_NAMES.keys():
        get_instance_predictions_and_embeddings_segpath(path=ROOT, cell_type=cell_type, n_images=args.n_images,
                                                        maximize_instances=False, min_instances=5)


if __name__ == "__main__":
    main()
