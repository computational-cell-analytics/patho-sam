import itertools
import json
from collections import defaultdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from organize_semantic_dataset import LABEL_MAP, LABELS, ROOT
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from util import get_rf_data_cancerscout

# v1 param grid
#  PARAM_GRID = {
#         "n_estimators": [200],
#         "max_depth": [None, 20],
#         "min_samples_leaf": [1, 5],
#         "max_features": ["sqrt", 0.3],
#         "class_weight": [None, "balanced"],
#     }
#  --> best was taken

# v2 param grid
PARAM_GRID = {
    "n_estimators": [200],
    "max_depth": [None],
    "min_samples_leaf": [3, 5, 10],
    "max_features": ["sqrt", 0.3],
    "class_weight": ["balanced", None],
}

CSV_PATH = ROOT / "cancerscout_metadata" / "cancerscout_semantic_organized.csv"


PRIORITY_CLASSES = ["tumor_cells", "lymphocytes", "epithelial_cells"]


# def priority_ap_score(y_true, y_proba, priority_classes: List, priority_indices):
#     """This scoring function optimizes the grid search for a balanced precision among the priority classes, namely
#     lymphocytes and epithelial nuclei"""
#     aps = []

#     for cls in priority_classes:
#         y_bin = (y_true == cls).astype(int)
#         aps.append(average_precision_score(y_bin, y_proba[:, cls]))

#     # We get the minimal average precision in order to not sacrifice one of the priority classes

#     return np.min(aps)


def priority_ap_scorer(priority_classes):
    def score(estimator, X, y_true):
        y_proba = estimator.predict_proba(X)

        classes = estimator.classes_
        priority_indices = [np.where(classes == c)[0][0] for c in priority_classes]

        aps = []
        for cls, idx in zip(priority_classes, priority_indices):
            y_bin = (y_true == cls).astype(int)
            aps.append(average_precision_score(y_bin, y_proba[:, idx]))

        return np.min(aps)

    return score


def rf_grid_search(
    path: Path = ROOT,
    csv_path: Path = CSV_PATH,
    cell_types: List = list(LABELS.values()),
    collapse_others: bool = True,
    priority_class_names: List = PRIORITY_CLASSES,
    include_healthy: bool = True,
):
    rf_dir = ROOT / "rf_models"
    rf_dir.mkdir(exist_ok=True, parents=True)

    for include_healthy, collapse_others in itertools.product([True, False], [True, False]):
        features_train_split, labels_train_split = get_rf_data_cancerscout(
            csv_path, data_path=path, cell_types=cell_types, split="train", healthy=True
        )

        priority_ids = [LABELS[cls] for cls in priority_class_names]

        if not include_healthy:
            priority_ids.remove(LABELS["epithelial_cells"])

        if collapse_others:
            labels_train_split[~np.isin(labels_train_split, priority_ids)] = 4
            if not len(np.unique(labels_train_split)) == len(priority_ids) + 1:
                raise ValueError(f"Label collapse not successful. {len(np.unique(labels_train_split))}, {priority_ids}")

        train_features, cal_features, train_labels, cal_labels = train_test_split(
            features_train_split, labels_train_split, test_size=0.2, random_state=0, stratify=labels_train_split
        )

        train_features, thresh_features, train_labels, thresh_labels = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=0, stratify=train_labels
        )

        rf = RandomForestClassifier(random_state=0, n_jobs=1)

        # _priority_ap_score = partial(priority_ap_score, priority_classes=priority_ids)

        # priority_scorer = make_scorer(_priority_ap_score, response_method="predict_proba")

        priority_scorer = priority_ap_scorer(priority_ids)

        grid = GridSearchCV(
            rf,
            PARAM_GRID,
            cv=5,
            scoring=priority_scorer,  # or "average_precision" if imbalanced
            verbose=2,
            refit=True,
            n_jobs=cpu_count() - 2,
        )

        grid.fit(train_features, train_labels)
        print("Grid search complete.")

        best_rf = grid.best_estimator_

        print(grid.best_params_)

        grid_specs = ""
        if not include_healthy:
            grid_specs += "_exclude_healthy"
        if collapse_others:
            grid_specs += "_collapse_classes"

        with open(rf_dir / f"best_params_{grid_specs}.json", "w") as f:
            json.dump(best_rf.get_params(), f, indent=4)

        for method in ["sigmoid", "isotonic"]:
            calibrator = CalibratedClassifierCV(best_rf, method=method, cv="prefit")

            calibrator.fit(cal_features, cal_labels)
            specs = grid_specs + "_" + method

            joblib.dump(calibrator, (rf_dir / f"calibrated_{specs}.joblib"))
            classes = calibrator.classes_

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
                df.to_csv(rf_dir / f"results_{specs}_{LABEL_MAP[ncl_cls]}_.csv")


def main():
    rf_grid_search()


if __name__ == "__main__":
    main()
