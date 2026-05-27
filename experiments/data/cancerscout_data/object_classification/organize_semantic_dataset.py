import json
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

LABELS = {"tumor_cells": 1, "stromal_cells": 2, "lymphocytes": 3, "others": 4, "neutrophils": 5, "epithelial_cells": 6}

LABEL_MAP = {v: k for k, v in LABELS.items()}

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data")


def organize_cancerscout():
    cs_metadata_csv = ROOT / "cancerscout_metadata" / "cancerscout_organized.csv"

    df = pd.read_csv(cs_metadata_csv, index_col="patient ID")

    csv_outpath = ROOT / "cancerscout_metadata" / "cancerscout_semantic_organized.csv"
    dfs = []
    for split in ["train", "eval"]:
        h5_paths = sorted(list(ROOT.glob(f"{split}_models/*data/fixed_h5_files/*.h5")))
        json_path = ROOT / f"{split}_models" / f"{split}_rois.json"
        with open(json_path, "r") as f:
            data = json.load(f)
        split_ids = []
        split_ids.extend([list(value.keys()) for key, value in data.items() if key != "pdl1_ihc"])
        split_ids = [int(n.split("-")[1].split("_")[0][2:]) for n in list(chain.from_iterable(split_ids))]
        split_df = df.loc[split_ids]
        for h5_path in h5_paths:
            patient_id = int(h5_path.stem.split("-")[1].split("_")[0][2:])

            assert patient_id in split_df.index.tolist(), patient_id
            with h5py.File(h5_path, "r") as f:
                total_inst_count = f["all_features"].shape[0]
                semantic_labels = f["train_labels"][:]

            nuclei_classes, counts = np.unique(semantic_labels, return_counts=True)
            sd = {cls: int(count) for cls, count in zip(nuclei_classes, counts)}
            split_df.loc[patient_id, "num_instances"] = total_inst_count
            for nucleus_class, name in LABEL_MAP.items():
                split_df.loc[patient_id, name] = int(sd.get(nucleus_class, 0))
            split_df.loc[patient_id, "filename"] = h5_path.stem
        split_df[[v for v in LABEL_MAP.values()]] = split_df[[v for v in LABEL_MAP.values()]].astype("Int64")
        split_df["inst_dataset"] = df["inst_dataset"].fillna("new_tumor")
        dfs.append(split_df)
    output_df = pd.concat(dfs, axis=0)
    output_df.to_csv(csv_outpath)
