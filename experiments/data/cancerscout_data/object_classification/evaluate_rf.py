import json
from pathlib import Path

import joblib
import numpy as np
from rf_grid_search import CSV_PATH, ROOT
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from util import CELL_ID_NAME, CELL_NAME_ID, SUBTYPE_ID_NAME, get_rf_data_cancerscout

RESULT_DIR = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/rf_models")


def evaluate_rf(data_path: Path, csv_path: Path, cell_types: dict, rf_dir: Path):
    reports_dir = RESULT_DIR / "classification_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    threshold_json = rf_dir / "all_thresholds.json"
    with open(threshold_json, "r") as f:
        thresholds = json.load(f)

    eval_features, eval_labels, eval_subtypes = get_rf_data_cancerscout(
        csv_path=csv_path,
        data_path=data_path,
        cell_types=cell_types.values(),
        healthy=True,
        split="eval",
        return_subtypes=True,
    )
    rf_paths = list(rf_dir.glob("models/*.joblib"))
    for rf_path in rf_paths:
        config_name = rf_path.stem.split("__")[-1]

        conf_thresholds: dict = thresholds[config_name]
        rf: RandomForestClassifier = joblib.load(rf_path)
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

        overall_report = classification_report(eval_labels, mapped_pred, output_dict=True, zero_division=0)
        with open(reports_dir / f"{config_name}_overall.json", "w") as f:
            json.dump(overall_report, f, indent=4)

        for subtype in np.unique(eval_subtypes).tolist():
            mask = eval_subtypes == subtype
            subtype_pred = mapped_pred[mask]
            subtype_labels = eval_labels[mask]
            subtype_report = classification_report(subtype_labels, subtype_pred, output_dict=True, zero_division=0)
            with open(reports_dir / f"{config_name}_{SUBTYPE_ID_NAME[subtype]}.json", "w") as f:
                json.dump(subtype_report, f, indent=4)


evaluate_rf(csv_path=CSV_PATH, data_path=ROOT, cell_types=CELL_ID_NAME, rf_dir=ROOT / "rf_models")
