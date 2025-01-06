import os
import shutil
import zipfile
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from elf.evaluation import mean_segmentation_accuracy
from natsort import natsorted
from skimage.measure import label
from tqdm import tqdm


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
    print(len(gt_paths), len(prediction_paths))
    assert len(gt_paths) == len(
        prediction_paths
    ), f"label / prediction mismatch: {len(gt_paths)} / {len(prediction_paths)}"
    msas, sa50s, sa75s = [], [], []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths, strict=False),
        desc="Evaluate predictions",
        total=len(gt_paths),
        disable=not verbose,
    ):
        assert os.path.exists(gt_path), gt_path
        assert os.path.exists(pred_path), pred_path

        gt = imageio.imread(gt_path)
        gt = label(gt)
        pred = imageio.imread(pred_path)

        msa, scores = mean_segmentation_accuracy(pred, gt, return_accuracies=True)
        sa50, sa75 = scores[0], scores[5]
        msas.append(msa), sa50s.append(sa50), sa75s.append(sa75)

    return msas, sa50s, sa75s


def evaluate_all_datasets_cellvit(prediction_dir, result_dir):
    for dataset in [
        "cpm15",
        "cpm17",
        "cryonuseg",
        "janowczyk",
        "lizard",
        "lynsec",
        "monusac",
        "monuseg",
        "nuinsseg",
        "pannuke",
        "puma",
        "tnbc",
    ]:
        for checkpoint in ["256-x20", "256-x40", "SAM-H-x20", "SAM-H-x40"]:
            save_path = os.path.join(result_dir, dataset, checkpoint, "ais_result.csv")
            if os.path.exists(save_path):
                continue
            with zipfile.ZipFile(os.path.join(prediction_dir, dataset, f"{checkpoint}.zip"), "r") as zipf:
                checkpoint_dir = os.path.join(prediction_dir, dataset, checkpoint)
                os.makedirs(checkpoint_dir)
                zipf.extractall(checkpoint_dir)
            prediction_paths = natsorted(glob(os.path.join(checkpoint_dir, "predictions", "*.tiff")))
            gt_paths = natsorted(glob(os.path.join(checkpoint_dir, "labels", "*.tiff")))
            if len(prediction_paths) == 0:
                print(f"No predictions for {dataset} dataset on {checkpoint} checkpoint found")
                continue
            msas, sa50s, sa75s = _run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths)
            results = pd.DataFrame.from_dict(
                {
                    "mSA": [np.mean(msas)],
                    "SA50": [np.mean(sa50s)],
                    "SA75": [np.mean(sa75s)],
                }
            )
            os.makedirs(os.path.join(result_dir, dataset, checkpoint), exist_ok=True)
            results.to_csv(save_path, index=False)
            shutil.rmtree(checkpoint_dir)

def main():
    evaluate_all_datasets_cellvit(
    "/mnt/lustre-grete/usr/u12649/models/cellvit/inference",
    "/mnt/lustre-grete/usr/u12649/models/cellvit/results",
    )   


if __name__ == "__main__":
    main()
