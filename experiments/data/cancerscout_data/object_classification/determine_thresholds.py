import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULT_DIR = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/rf_models")
CELL_NAMES = sorted(["tumor_cells", "lymphocytes", "stromal_cells", "epithelial_cells"], key=len, reverse=True)
pattern = r"(?:_|^)(" + "|".join(CELL_NAMES) + r")(?:_|$)"


def determine_model_thresholds(target_precision):
    all_result_csvs = list(RESULT_DIR.glob("*.csv"))
    all_thresholds = {}
    for include_healthy, collapse_others, method in itertools.product(
        ["exclude_healthy_", "_"], ["collapse_classes_", "_"], ["sigmoid", "isotonic"]
    ):
        results = {}

        substring = include_healthy + collapse_others + method
        substring = substring.replace("__", "_")
        # print(substring)
        # breakpoint()
        class_paths = [p for p in all_result_csvs if substring in p.name]
        for class_csv in class_paths:
            cell_type = class_csv.stem.strip("results__")
            # breakpoint()
            cell_type = re.search(pattern=pattern, string=class_csv.stem).group(1)

            df = pd.read_csv(class_csv)
            threshold, best_row = select_threshold_from_constraint(df, target_precision)

            if best_row is None:
                print(f"{cell_type}: constraint not reachable")
                continue

            results[cell_type] = {
                "threshold": threshold,
                "precision": best_row["precision"],
                "recall": best_row["recall"],
            }

            print(f"{cell_type} | t={threshold:.4f} | P={best_row['precision']:.4f} | R={best_row['recall']:.4f}")

        all_thresholds[substring.strip("_")] = results

    with open(RESULT_DIR / "all_thresholds.json", "w") as f:
        json.dump(all_thresholds, f, indent=4)


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


determine_model_thresholds(target_precision=0.85)
