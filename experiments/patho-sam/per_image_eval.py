import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import List, Optional, Union

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
    "monusac",
    "monuseg",
    "nuclick",
    "nuinsseg",
    "pannuke",
    "puma",
    "srsanet",
    "tnbc",
]

SAM_MODELS = ['generalist_sam', 'pannuke_sam']

MODELS = [
    'stardist',
    'hovernet',
    'hovernext',
    'instanseg',
    'cellvit',
    # 'cellvitpp',
    'generalist_sam',
    'pannuke_sam',
    # 'old_generalist_sam'
]

MODEL_NAMES = {
    'stardist': "StarDist",
    'hovernet': "HoVerNet",
    'hovernext': "HoVerNeXt",
    'instanseg': "InstanSeg",
    'cellvit': "CellViT",
    'pannuke_sam': "Patho-SAM (Specialist)",
    'generalist_sam': "Patho-SAM (Generalist)",
}

HNXT_CP = [
    "lizard_convnextv2_large",
    "lizard_convnextv2_base",
    "lizard_convnextv2_tiny",
    "pannuke_convnextv2_tiny_1",
    "pannuke_convnextv2_tiny_2",
    "pannuke_convnextv2_tiny_3",
]
CVT_CP = [
    "256-x20",
    "256-x40",
    "SAM-H-x20",
    "SAM-H-x40",
]

CVTPP_CP = ["SAM-H-x40-AMP"]

SAM_TYPES = ["vit_b", "vit_l", "vit_h"]


HVNT_CP = [
    'consep',
    'cpm17',
    'kumar',
    'monusac',
    'pannuke',
]

CHECKPOINTS = {
    'hovernet': HVNT_CP,
    'hovernext': HNXT_CP,
    'cellvit': CVT_CP,
    'cellvitpp': CVTPP_CP,
    'generalist_sam': SAM_TYPES,
    'pannuke_sam': ['vit_b'],
    'old_generalist_sam': ['vit_b'],
    'stardist': ['stardist'],
    'instanseg': ['instanseg'],
}


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
    assert len(gt_paths) == len(prediction_paths), f"{len(gt_paths)}, {len(prediction_paths)}"
    msas, sa50s, sa75s = [], [], []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths), desc="Evaluate predictions", total=len(gt_paths), disable=not verbose
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


def run_evaluation(
    gt_paths: List[Union[os.PathLike, str]],
    prediction_paths: List[Union[os.PathLike, str]],
    save_path: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run evaluation for instance segmentation predictions.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_paths: The list of paths with the instance segmentations to evaluate.
        save_path: Optional path for saving the results.
        verbose: Whether to print the progress.

    Returns:
        A DataFrame that contains the evaluation results.
    """
    assert len(gt_paths) == len(prediction_paths), f"{len(gt_paths)}, {len(prediction_paths)}"
    # if a save_path is given and it already exists then just load it instead of running the eval
    if save_path is not None and os.path.exists(save_path):
        return pd.read_csv(save_path)

    msas, sa50s, sa75s = _run_evaluation(gt_paths, prediction_paths, verbose=verbose)

    results = np.array(msas)
    np.save(save_path, results)


def per_sample_eval(inf_path, data_path):
    for model in MODELS:
        for checkpoint in CHECKPOINTS[model]:
            for dataset in DATASETS:
                if model not in SAM_MODELS:
                    inference_paths = natsorted(
                        glob(os.path.join(inf_path, model, "inference", dataset, checkpoint, "*.tiff"))
                    )
                else:
                    inference_paths = natsorted(
                        glob(
                            os.path.join(
                                inf_path, model, "inference", dataset, checkpoint, "instance",
                                "instance_segmentation_with_decoder", "inference", "*.tiff"
                            )
                        )
                    )

                label_paths = natsorted(
                    glob(os.path.join(data_path, dataset, "loaded_testset", "eval_split", "test_labels", "*.tiff"))
                )
                save_path = os.path.join(
                    inf_path, model, "results", "per_sample", f"{model}_{dataset}_{checkpoint}_ps.npy"
                )
                if os.path.exists(save_path):
                    continue

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"evaluating {model}, checkpoint {checkpoint}")
                run_evaluation(gt_paths=label_paths, prediction_paths=inference_paths, save_path=save_path)


per_sample_eval(
    data_path="/mnt/lustre-grete/usr/u12649/data/final_test",
    inf_path="/mnt/lustre-grete/usr/u12649/models",
)
