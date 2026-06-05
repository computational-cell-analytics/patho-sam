from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data")

CELL_NAME_ID = {
    "tumor_cells": 1,
    "stromal_cells": 2,
    "lymphocytes": 3,
    "others": 4,
    "neutrophils": 5,
    "epithelial_cells": 6,
}

SUBTYPE_NAME_ID = {
    "acinous": 1,
    "papillary": 2,
    "solid": 3,
    "mucinous": 4,
    "lepidic": 5,
    # "lepidic_mixed": 6,
    "micropapillary": 7,
}

SUBTYPE_ID_NAME = {v: k for k, v in SUBTYPE_NAME_ID.items()}
CELL_ID_NAME = {v: k for k, v in CELL_NAME_ID.items()}


def get_rf_data_cancerscout(
    csv_path: Path,
    data_path: Path,
    split: str,
    healthy: bool,
    cell_types: List,
    return_subtypes: bool = False,
    subtype: str = None,
) -> Tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(csv_path, index_col="filename")

    df = df[df["train_eval_split"] == split]
    if not healthy:
        df = df[df["inst_dataset"] == "new_tumor"]

    if subtype:
        df = df[df["subtype"] == subtype]

    if df.empty:
        print(f"No training data for subtype {subtype}")
        return None, None

    num_training_objects = df[cell_types].sum(axis=1).sum()
    cell_types = [CELL_NAME_ID[key] for key in cell_types]

    chosen_samples = df.index.tolist()

    all_h5_paths = list(data_path.glob(f"{split}_models/*data/fixed_h5_files/*.h5"))

    chosen_h5_paths = [p for p in all_h5_paths if p.stem in chosen_samples]
    offset = 0
    with h5py.File(chosen_h5_paths[0], "r") as f:
        feature_dim = f["train_features"].shape[1]
        feature_dtype = f["train_features"].dtype

    features_out = np.empty((num_training_objects, feature_dim), dtype=feature_dtype)
    labels_out = np.empty((num_training_objects), dtype=np.uint8)
    subtypes_out = np.empty((num_training_objects), dtype=np.uint8)

    for volume_path in chosen_h5_paths:
        with h5py.File(volume_path, "r") as f:
            subtype = SUBTYPE_NAME_ID[df.loc[volume_path.stem, "subtype"]]
            features = f["train_features"]
            labels = f["train_labels"]
            n = features.shape[0]
            features_out[offset : offset + n] = features
            labels_out[offset : offset + n] = labels
            subtypes_out[offset : offset + n] = subtype
            offset += n

    if return_subtypes:
        return features_out, labels_out, subtypes_out
    else:
        return features_out, labels_out


def determine_model_thresholds(target_precision, df, cell_type):
    threshold, best_row = select_threshold_from_constraint(df, target_precision)

    if best_row is None:
        print(f"{cell_type}: constraint not reachable")
        return

    return {"threshold": threshold, "precision": best_row["precision"], "recall": best_row["recall"]}


def get_pareto_front(df):
    """
    Returns Pareto-optimal points in (precision, recall)-space.
    Assumes higher precision and higher recall are better.
    """
    # sort for stable processing
    df = df.sort_values(["recall", "precision"], ascending=[False, False]).copy()

    pareto = []
    best_precision = -np.inf

    # iterate by decreasing recall
    for _, row in df.iterrows():
        if row["precision"] > best_precision:
            pareto.append(row)
            best_precision = row["precision"]

    return pd.DataFrame(pareto)


def select_threshold_from_constraint(df, target_precision):
    """
    Select threshold using:
    1. precision constraint
    2. Pareto-optimal filtering
    3. maximize recall
    4. tie-breaker: max precision
    """

    df = df.copy()

    # enforce constraint
    constrained = df[df["precision"] >= target_precision]

    if constrained.empty:
        return None, None

    # pareto filter within constrained set
    pareto = get_pareto_front(constrained)

    # best recall on pareto frontier
    max_recall = pareto["recall"].max()
    candidates = pareto[pareto["recall"] == max_recall]

    # tie-break: best precision
    best_row = candidates.loc[candidates["precision"].idxmax()]

    return best_row["thresholds"], best_row


def evaluate_rf(eval_features: np.ndarray, eval_labels: np.ndarray, conf_thresholds: dict, rf: RandomForestClassifier):
    probs = rf.predict_proba(eval_features)

    n_samples, n_classes = probs.shape
    pred = np.argmax(probs, axis=1)
    class_to_idx = {cls_id: i for i, cls_id in enumerate(rf.classes_)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for j in range(n_samples):
        candidates = []

        for cls_name, t in conf_thresholds.items():
            cell_id = CELL_NAME_ID[cls_name]
            i = class_to_idx[cell_id]

            if probs[j, i] >= t["threshold"]:
                candidates.append(i)

        if len(candidates) > 0:
            pred[j] = max(candidates, key=lambda i: probs[j, i])
    mapped_pred = np.array([idx_to_class[i] for i in pred])

    return classification_report(eval_labels, mapped_pred, output_dict=True, zero_division=0)
