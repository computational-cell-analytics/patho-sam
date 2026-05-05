from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import numpy as np
from assign_splits import create_data_splits
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from train_rf_segpath import get_rf_data
from util import CELL_NAMES, ROOT

PARAM_GRID = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True],
}


def rf_grid_search(
    path: Path = ROOT,
    cell_types: List = list(CELL_NAMES.values()),
    n_samples: int = None,
    collapse_others: bool = False,
):

    create_data_splits(path, overwrite_split=True, n_samples=n_samples)

    features_train_split, labels_train_split = get_rf_data(ROOT, cell_types=cell_types, split="train")

    if collapse_others:
        labels_train_split[~np.isin(labels_train_split, [1, 2])] = 3
        if not len(np.unique(labels_train_split)) == 3:
            raise ValueError("Label collapse not successfull, np unique does not equal 3 as expected")

    train_features, cal_features, train_labels, cal_labels = train_test_split(
        features_train_split, labels_train_split, test_size=0.2, random_state=0, stratify=labels_train_split
    )

    rf = RandomForestClassifier(random_state=0, n_jobs=cpu_count() - 1)

    grid = GridSearchCV(
        rf,
        PARAM_GRID,
        cv=5,
        scoring="average_precision",  # or "average_precision" if imbalanced
        verbose=2,
        refit=True,
    )

    grid.fit(train_features, train_labels)

    best_rf = grid.best_estimator_

    calibrator = CalibratedClassifierCV(best_rf, method="isotonic", cv="prefit")

    calibrator.fit(cal_features, cal_labels)


def main():
    rf_grid_search()


if __name__ == "__main__":
    main()
