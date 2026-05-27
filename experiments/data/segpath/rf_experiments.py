from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from assign_splits import create_data_splits
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from train_rf_segpath import get_rf_data
from util import CELL_NAMES, CELL_TYPE_MAPPING, ROOT

PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.3],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "bootstrap": [True],
}

PRIORITY_CLASSES = [1, 2]


def priority_ap_score(y_true, y_proba, priority_classes: List):
    """This scoring function optimizes the grid search for a balanced precision among the priority classes, namely
    lymphocytes and epithelial nuclei"""
    aps = []

    for cls in priority_classes:
        y_bin = (y_true == cls).astype(int)
        aps.append(average_precision_score(y_bin, y_proba[:, cls]))

    # We get the minimal average precision in order to not sacrifice one of the priority classes

    return np.min(aps)


def rf_grid_search(
    path: Path = ROOT,
    cell_types: List = list(CELL_NAMES.values()),
    n_samples: int = None,
    collapse_others: bool = True,
    priority_classes: List = PRIORITY_CLASSES,
):
    for cell_type in cell_types:
        create_data_splits(path, overwrite_split=True, n_samples=n_samples, cell_type=cell_type)

    features_train_split, labels_train_split = get_rf_data(ROOT, cell_types=cell_types, split="train")

    if collapse_others:
        labels_train_split[~np.isin(labels_train_split, [1, 2])] = 3
        if not len(np.unique(labels_train_split)) == 3:
            raise ValueError("Label collapse not successfull, np unique does not equal 3 as expected")

    train_features, cal_features, train_labels, cal_labels = train_test_split(
        features_train_split, labels_train_split, test_size=0.2, random_state=0, stratify=labels_train_split
    )

    train_features, thresh_features, train_labels, thresh_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=0, stratify=train_labels
    )

    rf = RandomForestClassifier(random_state=0, n_jobs=1)

    _priority_ap_score = partial(priority_ap_score, priority_classes=priority_classes)

    priority_scorer = make_scorer(_priority_ap_score, response_method="predict_proba")

    grid = GridSearchCV(
        rf,
        PARAM_GRID,
        cv=5,
        scoring=priority_scorer,  # or "average_precision" if imbalanced
        verbose=2,
        refit=True,
        n_jobs=cpu_count() - 1,
    )

    grid.fit(train_features, train_labels)

    best_rf = grid.best_estimator_

    for method in ["sigmoid", "isotinic"]:
        calibrator = CalibratedClassifierCV(best_rf, method=method, cv="prefit")

        calibrator.fit(cal_features, cal_labels)

        joblib.dump(calibrator, (ROOT / "rf_models" / f"calibrated_{method}_rf.joblib"))

        for ncl_cls in priority_classes:
            result_dict = {}

            probs = calibrator.predict_proba(thresh_features)
            p_c = probs[:, ncl_cls]
            y_c = (thresh_labels == ncl_cls).astype(int)

            precisions = []
            recalls = []
            f1s = []
            thresholds = np.linspace(0, 1, 200)
            for t in thresholds:
                y_pred = (p_c >= t).astype(int)

                prec = precision_score(y_c, y_pred, zero_division=0)
                rec = recall_score(y_c, y_pred, zero_division=0)
                f1 = 2 * prec * rec / (prec + rec + 1e-12)

                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
            result_dict["thresholds"] = thresholds
            result_dict["precision"] = precisions
            result_dict["recall"] = recalls
            result_dict["f1"] = f1s
            df = pd.DataFrame(result_dict)
            df.to_csv(ROOT / "rf_models" / f"results_{method}_{CELL_TYPE_MAPPING[ncl_cls]}.csv")


def main():
    rf_grid_search()


if __name__ == "__main__":
    main()
