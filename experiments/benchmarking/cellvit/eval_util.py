import os
import zipfile
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import label

from elf.evaluation import mean_segmentation_accuracy


def zip_predictions(path, target_dir):
    print(f"Zipping {path}...")
    zip_name = os.path.basename(path) + ".zip"
    zip_path = os.path.join(target_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=path)
                zipf.write(file_path, arcname)
    print("Successfully zipped results")


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
    print(len(gt_paths), len(prediction_paths))
    assert len(gt_paths) == len(prediction_paths), \
        f"label / prediction mismatch: {len(gt_paths)} / {len(prediction_paths)}"

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


def evaluate_cellvit(prediction_dir, checkpoint, dataset, label_dir, result_dir):
    save_path = os.path.join(dataset, checkpoint, f'{dataset}_cellvit_{checkpoint}_ais_result.csv')
    if os.path.exists(save_path):
        print(f"Results for {dataset} evaluation already exist")
        return
    prediction_paths = natsorted(glob(os.path.join(prediction_dir, "*")))
    gt_paths = natsorted(glob(os.path.join(label_dir, "test_labels", "*")))
    if len(prediction_paths) == 0:
        print(f"No predictions for {dataset} dataset on {checkpoint} checkpoint found")
        return

    msas, sa50s, sa75s = _run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths)
    results = pd.DataFrame.from_dict(
        {
            "mSA": [np.mean(msas)], "SA50": [np.mean(sa50s)], "SA75": [np.mean(sa75s)],
        }
    )
    print(results.head(2))
    os.makedirs(os.path.join(result_dir, dataset, checkpoint), exist_ok=True)
    results.to_csv(save_path, index=False)
