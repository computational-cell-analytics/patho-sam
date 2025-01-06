import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from elf.evaluation import mean_segmentation_accuracy
from natsort import natsorted
from skimage.measure import label
from tqdm import tqdm

CHECKPOINTS = [
    "lizard_convnextv2_large",
    "lizard_convnextv2_base",
    "lizard_convnextv2_tiny",
    "pannuke_convnextv2_tiny_1",
    "pannuke_convnextv2_tiny_2",
    "pannuke_convnextv2_tiny_3",
]


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
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


def evaluate_all_datasets_hovernet(prediction_dir, label_dir, result_dir):
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
        gt_paths = natsorted(
            glob(os.path.join(label_dir, dataset, "loaded_dataset/complete_dataset/eval_split/test_labels/*.tiff"))
        )
        for checkpoint in CHECKPOINTS:
            save_path = os.path.join(result_dir, dataset, checkpoint, "ais_result.csv")
            if os.path.exists(save_path):
                continue
            prediction_paths = natsorted(glob(os.path.join(prediction_dir, dataset, checkpoint, "*.tiff")))
            if len(prediction_paths) == 0:
                print(f"No predictions for {dataset} dataset on {checkpoint} checkpoint found")
                continue
            os.makedirs(os.path.join(result_dir, dataset, checkpoint), exist_ok=True)
            print(f"evaluation {dataset} dataset on checkpoint {checkpoint} ...")
            msas, sa50s, sa75s = _run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths)
            results = pd.DataFrame.from_dict(
                {
                    "mSA": [np.mean(msas)],
                    "SA50": [np.mean(sa50s)],
                    "SA75": [np.mean(sa75s)],
                }
            )

            results.to_csv(save_path, index=False)


evaluate_all_datasets_hovernet(
    "/mnt/lustre-grete/usr/u12649/models/hovernext/inference",
    "/mnt/lustre-grete/usr/u12649/data/test",
    "/mnt/lustre-grete/usr/u12649/models/hovernext/results",
)
