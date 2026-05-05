from pathlib import Path

CELL_NAMES = {
    "smooth_muscle": "aSMA_SmoothMuscle",
    "endothelium": "ERG_Endothelium",
    "lymphocytes": "CD3CD20_Lymphocyte",
    "epithelium": "panCK_Epithelium",
    "plasma_cells": "MIST1_PlasmaCell",
    "leukocytes": "CD45RB_Leukocyte",
}

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/segpath")

CELL_TYPE_MAPPING = {
    "panCK_Epithelium": 1,
    "CD3CD20_Lymphocyte": 2,
    "aSMA_SmoothMuscle": 3,
    "ERG_Endothelium": 4,
    "CD45RB_Leukocyte": 5,
    "MIST1_PlasmaCell": 6,
}
