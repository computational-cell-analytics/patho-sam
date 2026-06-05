import itertools
import json
from collections import defaultdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from util import (
    CELL_ID_NAME,
    CELL_NAME_ID,
    ROOT,
    SUBTYPE_NAME_ID,
    determine_model_thresholds,
    evaluate_rf,
    get_rf_data_cancerscout,
)

CSV_PATH = ROOT / "cancerscout_metadata" / "cancerscout_semantic_organized.csv"

PARAMS_DIR = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/rf_models/params")

PRIORITY_CLASSES = ["tumor_cells", "lymphocytes", "epithelial_cells"]


def get_subtype_data(subtype_features, subtype_labels, collapse_others, priority_ids):

    if collapse_others:
        train_labels = subtype_labels[~np.isin(subtype_labels, priority_ids)] = 4

    train_features, cal_features, train_labels, cal_labels = train_test_split(
        subtype_features, subtype_labels, test_size=0.2, random_state=0, stratify=subtype_labels
    )

    train_features, thresh_features, train_labels, thresh_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=0, stratify=train_labels
    )

    return train_features, train_labels, cal_features, cal_labels, thresh_features, thresh_labels


def train_rf(
    path: Path = ROOT,
    csv_path: Path = CSV_PATH,
    cell_types: List = list(CELL_NAME_ID.keys()),
    priority_class_names: List = PRIORITY_CLASSES,
    calibration_method: str = "isotonic",
    target_precision: float = 0.85,
):
    rf_dir = ROOT / "rf_models" / "per_subtype_training"

    for subtype_name in tqdm(list(SUBTYPE_NAME_ID.keys())):
        subtype_dir = rf_dir / subtype_name
        subtype_dir.mkdir(exist_ok=True, parents=True)

        eval_features, eval_labels = get_rf_data_cancerscout(
            csv_path=csv_path, data_path=path, cell_types=cell_types, healthy=True, split="eval", subtype=subtype_name
        )

        subtype_features, subtype_labels = get_rf_data_cancerscout(
            csv_path, data_path=path, cell_types=cell_types, split="train", healthy=True, subtype=subtype_name
        )

        if subtype_features is None:
            continue

        for include_healthy, collapse_others in itertools.product([True, False], [True, False]):
            priority_ids = [CELL_NAME_ID[cell_name] for cell_name in priority_class_names]
            if not include_healthy:
                priority_ids.remove(CELL_NAME_ID["epithelial_cells"])

            train_features, train_labels, cal_features, cal_labels, thresh_features, thresh_labels = get_subtype_data(
                subtype_features, subtype_labels, collapse_others, priority_ids
            )

            if CELL_NAME_ID["epithelial_cells"] in priority_ids and CELL_NAME_ID["epithelial_cells"] not in np.unique(
                train_labels
            ):
                priority_ids.remove(CELL_NAME_ID["epithelial_cells"])

            params_spec = ""
            if not include_healthy:
                params_spec += "_exclude_healthy"
            if collapse_others:
                params_spec += "_collapse_classes"

            results_json_path = subtype_dir / f"classification_results{params_spec}.json"
            if results_json_path.exists():
                continue

            params_path = PARAMS_DIR / f"best_params_{params_spec}.json"
            assert params_path.exists(), params_path
            with open(params_path, "r") as f:
                params = json.load(f)

            params["n_jobs"] = cpu_count() - 2

            rf = RandomForestClassifier(**params)

            rf.fit(train_features, train_labels)
            print("RF training complete.")

            calibrator = CalibratedClassifierCV(rf, method=calibration_method, cv="prefit")

            calibrator.fit(cal_features, cal_labels)
            specs = params_spec + "_" + calibration_method

            classes = calibrator.classes_

            threshold_dict = {}

            for ncl_cls in priority_ids:
                result_dict = defaultdict(list)
                idx = np.where(classes == ncl_cls)[0][0]
                probs = calibrator.predict_proba(thresh_features)
                p_c = probs[:, idx]
                y_c = (thresh_labels == ncl_cls).astype(int)

                thresholds = np.linspace(0, 1, 200)
                result_dict["thresholds"] = thresholds
                for t in thresholds:
                    y_pred = (p_c >= t).astype(int)

                    prec = precision_score(y_c, y_pred, zero_division=0)
                    rec = recall_score(y_c, y_pred, zero_division=0)
                    f1 = 2 * prec * rec / (prec + rec + 1e-12)

                    result_dict["precision"].append(prec)
                    result_dict["recall"].append(rec)
                    result_dict["f1"].append(f1)

                df = pd.DataFrame(result_dict)

                df.to_csv(subtype_dir / f"results_{specs}_{CELL_ID_NAME[ncl_cls]}_.csv")
                threshold_dict[CELL_ID_NAME[ncl_cls]] = determine_model_thresholds(
                    target_precision, df, cell_type=CELL_ID_NAME[ncl_cls]
                )

            classification_report = evaluate_rf(eval_features, eval_labels, threshold_dict, calibrator)
            with open(results_json_path, "w") as f:
                json.dump(classification_report, f, indent=4)


def main():
    train_rf()


if __name__ == "__main__":
    main()
