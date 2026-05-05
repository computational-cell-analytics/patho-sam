import argparse
from pathlib import Path

import h5py
import pandas as pd
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from tqdm import tqdm
from util import CELL_NAMES, ROOT


def get_instance_predictions_and_embeddings_segpath(
    path=ROOT, cell_type: str = None, n_images=None, maximize_instances: bool = False, min_instances: int = 0
):
    cell_type = CELL_NAMES[cell_type]
    data_path = Path(path) / cell_type / "data"
    embedding_dir = Path(path) / cell_type / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    df = pd.read_csv(Path(ROOT) / cell_type / f"{cell_type}_summary.csv", index_col="filename")

    # We only predict for images where binary masks exist for specified amount of minimum instances
    filtered_df = df[df["num_instances"] >= min_instances]

    if "randomly_sampled" not in df.keys():
        df["randomly_sampled"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    if maximize_instances:
        all_sampled_df = filtered_df.nlargest(n_images, "num_instances")

    else:
        all_sampled_df = filtered_df.sample(len(filtered_df.index), random_state=42)

    sample_df = all_sampled_df.iloc[:n_images]

    predict_samples = sample_df.index.tolist()

    df.loc[predict_samples, "randomly_sampled"] = not maximize_instances

    # Remove already predicted samples from predict_samples so we don't unnnecessarily access the .h5 file
    filtered_df = filtered_df[~filtered_df["predicted"]]

    volume_paths = [
        data_path / predict_sample for predict_sample in predict_samples if predict_sample in filtered_df.index
    ]

    predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_histopathology", segmentation_mode="apg")

    for volume_path in tqdm(volume_paths):
        embedding_path = embedding_dir / volume_path.stem

        with h5py.File(volume_path, "a") as f:
            if "labels/postprocessed_pred" in f:
                continue

            if "labels/raw_pred" not in f:
                segmentation = automatic_instance_segmentation(
                    predictor,
                    segmenter,
                    input_path=volume_path,
                    key="images/best_crop",
                    embedding_path=embedding_path,
                    ndim=2,
                    verbose=False,
                )
                f.create_dataset(name="labels/raw_pred", data=segmentation, compression="gzip")

    mapping = {path.name: True for path in volume_paths}
    df.loc[df.index.isin(mapping), "predicted"] = True
    df.to_csv(Path(ROOT) / cell_type / f"{cell_type}_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=ROOT)
    parser.add_argument("--cell_type", type=str, default=None)
    parser.add_argument("--n_images", type=int, default=500)
    args = parser.parse_args()
    if args.cell_type is None:
        for cell_type in CELL_NAMES.keys():
            get_instance_predictions_and_embeddings_segpath(
                path=args.input_path,
                cell_type=cell_type,
                n_images=args.n_images,
                maximize_instances=False,
                min_instances=3,
            )
    else:
        get_instance_predictions_and_embeddings_segpath(
            path=args.input_path, cell_type=cell_type, n_images=args.n_images, maximize_instances=False, min_instances=5
        )


if __name__ == "__main__":
    main()
