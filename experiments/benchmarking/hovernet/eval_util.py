import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import label

from elf.evaluation import mean_segmentation_accuracy


DATASETS = [
    "consep",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "lizard",
    "lynsec_he",
    "lynsec_ihc",
    "monuseg",
    "nuclick",
    "nuinsseg",
    "pannuke",
    "puma",
    "srsanet",
    "tnbc",
]


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
    assert len(gt_paths) == len(
        prediction_paths
    ), f"label / prediction mismatch: {len(gt_paths)} / {len(prediction_paths)}"
    msas, sa50s, sa75s = [], [], []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths),
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


def evaluate_hovernet(prediction_dir, label_dir, result_dir, dataset, checkpoint):
    gt_paths = natsorted(glob(os.path.join(label_dir, "*")))
    save_path = os.path.join(result_dir, dataset, checkpoint, f'{dataset}_hovernet_{checkpoint}_ais_result.csv')
    prediction_paths = natsorted(glob(os.path.join(prediction_dir, "*.tiff")))
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
    print(results.head())
    results.to_csv(save_path, index=False)
