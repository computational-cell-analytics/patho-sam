from pathlib import Path

CELL_NAMES = {"smooth_muscle": "aSMA_SmoothMuscle",
              "endothelium": "ERG_Endothelium",
              "lymphocytes": "CD3CD20_Lymphocyte",
              "epithelium": "panCK_Epithelium",
              "plasma_cells": "MIST1_PlasmaCell",
              "leukocytes": "CD45RB_Leukocyte",
              }

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data/segpath")

CELL_TYPE_MAPPING = {
    "epithelium": 1,
    "lymphocyte": 2,
    "smooth_muscle": 3,
    "endothelium": 4,
    "leukocytes": 5,
    "plasma_cells": 6
}
