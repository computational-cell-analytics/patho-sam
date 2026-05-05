from pathlib import Path
import h5py
import joblib
import json
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from typing import List, Dict
from functools import partial
from multiprocessing import cpu_count, Pool
from micro_sam.sam_annotator.object_classifier import _train_rf

from util import ROOT, CELL_TYPE_MAPPING, CELL_NAMES


def extract_from_h5(cell_type, split, path, df=None):
    if not df:
        df = pd.read_csv((path / cell_type / f"{cell_type}_summary.csv"), index_col="filename")
    
    filtered_df = df[(df["split"] == split)]
    filtered_df["filtered_indices"].apply(json.loads)
    num_training_objects = int(filtered_df["n_filtered_indices"].sum())
    volume_paths = [path / cell_type / "data" / filename for filename in filtered_df.index.tolist()]
    filtered_ids_list = filtered_df["filtered_indices"].tolist()
    offset = 0
    if not len(volume_paths) == len(filtered_ids_list):
        raise ValueError("Inconsistent length of input volume paths and filtered list indices")

    if not num_training_objects == sum(len(x) for x in filtered_ids_list):
        raise ValueError("Inconsistent number of df-provided training objects and filtered list indices")

    # Here we derive the feature dtype and dimensions from an arbitrary sample
    with h5py.File(volume_paths[0], 'r') as f:
        feature_dim = f["object_features"].shape[1]
        feature_dtype = f["object_features"].dtype

    out = np.empty((num_training_objects, feature_dim), dtype=feature_dtype)

    for volume_path, filtered_ids in zip(volume_paths, filtered_ids_list):
        with h5py.File(volume_path, 'r') as f:
            features = f["object_features"][:]
        n = len(filtered_ids)
        out[offset:offset+n] = features[filtered_ids]
        offset += n

    labels = np.full(out.shape[0], CELL_TYPE_MAPPING[cell_type])

    return out, labels


def get_rf_data(path: str, cell_types: List, split: str = None, df: pd.DataFrame = None):
    path = Path(path)

    all_features = []
    all_labels = []

    extract_from_h5_partial = partial(extract_from_h5, split=split, path=path, df=df)

    with Pool(cpu_count() - 2) as p:
        outputs = p.map(extract_from_h5_partial, cell_types)

    all_features = np.concatenate([output[0] for output in outputs], axis=0)
    all_labels = np.concatenate([output[1] for output in outputs], axis=0)

    return all_features, all_labels


def evaluate_rf(path: str, rf_path: str = None, cell_types: List = None, rf=None, cell_type_mapping: Dict = None,
                cell_name_mapping: Dict = None, df=None, test_features: np.ndarray = None, test_labels: np.ndarray = None):

    if test_features is None and test_labels is None:
        test_features, test_labels = get_rf_data(path, [cell_name_mapping[cell_type] for cell_type in cell_types],
                                                split="test", df=df)
    if not rf:
        rf = joblib.load(rf_path)

    pred = rf.predict(test_features)
    result_dict = classification_report(test_labels, pred, output_dict=True)
    for cell_type in cell_types:
        if str(cell_type_mapping[cell_name_mapping[cell_type]]) not in result_dict.keys():
            continue
        result_dict[cell_type] = result_dict.pop(str(cell_type_mapping[cell_name_mapping[cell_type]]))

    json_output_path = path / "rf_models" / "results" / f"evaluation_{rf_path.stem}.json"
    print(result_dict)
    with open(json_output_path, 'w') as f:
        json.dump(result_dict, f, indent=4)

    return result_dict


def train_rf(path, cell_types, cell_type_mapping=CELL_TYPE_MAPPING, cell_name_mapping=CELL_NAMES,
             df: pd.DataFrame = None, train_features: np.ndarray = None, train_labels: np.ndarray = None, 
             test_features: np.ndarray = None, test_labels: np.ndarray = None, **rf_kwargs):
    path = Path(path)

    rf_dir = path / "rf_models" / "models"
    rf_dir.mkdir(parents=True, exist_ok=True)
    rf_outpath = rf_dir / f"{'-'.join(cell_types)}.joblib"

    if train_features is None and train_labels is None:
        train_features, train_labels = get_rf_data(path, [CELL_NAMES[key] for key in cell_types], split="train", df=df)

    rf = _train_rf(
        features=train_features,
        labels=train_labels,
        n_estimators=200,
        max_depth=10,
        n_jobs=cpu_count(),
        **rf_kwargs
    )

    evaluate_rf(path, rf_outpath, cell_types, rf, cell_type_mapping, cell_name_mapping, df=df, test_features, test_labels)
    joblib.dump(rf, rf_outpath)


def main():
    train_rf(ROOT, ["epithelium", "lymphocytes", "smooth_muscle"], class_weight='balanced')


if __name__ == "__main__":
    main()
