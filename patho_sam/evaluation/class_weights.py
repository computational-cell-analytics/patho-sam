import os
from typing import List, Union
from glob import glob
import h5py
import numpy as np
import pandas as pd


LABEL_KEYS = {'pannuke': {'semantic': 'labels/semantic', 'instance': 'labels/instances'},
              'puma': {'semantic': 'labels/semantic/nuclei', 'instance': 'labels/instance/nuclei'},
              'conic': {'semantic': 'labels/semantic', 'instance': 'labels/instance'},
              }

CLASS_DICT = {
    'puma': {
        "nuclei_stroma": 1,
        "nuclei_tumor": 2,
        "nuclei_plasma_cell": 3,
        "nuclei_histiocyte": 4,
        "nuclei_lymphocyte": 5,
        "nuclei_melanophage": 6,
        "nuclei_neutrophil": 7,
        "nuclei_endothelium": 8,
        "nuclei_epithelium": 9,
        "nuclei_apoptosis": 10
    },
    'pannuke': {
        "neoplastic": 1,
        "inflammatory": 2,
        "connective / soft tissue": 3,
        "dead cells": 4,
        "epithelial": 5,
    },
    'conic': {
        "neutrophil": 1,
        "epithelial": 2,
        "lymphocyte": 3,
        "plasma": 4,
        "eosinophil": 5,
        "connective": 6,
    }
}


def extract_class_weights(fpath: Union[os.PathLike, str], dataset: str = None, output_path=None) -> List:
    """Extract class weights per semantic class.

    Args:
        fpath: The filepath where the input stack for PanNuke and CONIC are located. For PUMA, provide the
            path where .h5 files for the respective split are saved.
            Use `torch_em.data.datasets.histopathology.get_{pannuke, conic, puma}_paths` to get filepath for the stack.
        class_ids: The choice of all available class ids.
        dataset: The dataset to extract class weights for
    """
    # Load the entire instance and semantic stack.
    class_ids = CLASS_DICT[dataset].values()
    if dataset == 'puma':
        semantics = None
        instances = None
        for file in glob(os.path.join(fpath, "*.h5")):
            with h5py.File(file, "r") as f:
                instance = f[LABEL_KEYS[dataset]['instance']][:]
                semantic = f[LABEL_KEYS[dataset]['semantic']][:]
                if semantics is not None:
                    semantics = np.concatenate((semantics, semantic), axis=0)
                else:
                    semantics = semantic
                if instances is not None:
                    instances = np.concatenate((instances, instance), axis=0)
                else:
                    instances = instance
    else:
        with h5py.File(fpath, "r") as f:
            instances = f[LABEL_KEYS[dataset]['instance']][:]
            semantics = f[LABEL_KEYS[dataset]['semantic']][:]  
    # We need the following:
    # - Count the total number of instances.
    total_instance_counts = [
        len(np.unique(ilabel)[1:]) for ilabel in instances if len(np.unique(ilabel)) > 1
    ]  # Counting all valid foreground instances only.
    total_instance_counts = sum(total_instance_counts)

    # - Count per-semantic-class instances.
    total_per_class_instance_counts = [
        [len(np.unique(np.where(slabel == cid, ilabel, 0))[1:]) for cid in class_ids] # np.where --> insert instance ids for specific class instances, otherwise put 0. Outputs class-specific instance mask
        for ilabel, slabel in zip(instances, semantics) if len(np.unique(ilabel)) > 1
    ]
    assert total_instance_counts == sum([sum(t) for t in total_per_class_instance_counts])

    # Calculate per class mean values.
    total_per_class_instance_counts = [sum(x) for x in zip(*total_per_class_instance_counts)]
    assert total_instance_counts == sum(total_per_class_instance_counts)

    # Finally, let's get the weight per class. Results are saved as .csv in the output folder and output as a list
    per_class_weights = [t / total_instance_counts for t in total_per_class_instance_counts]
    result_dict = {nt: weight for nt, weight in zip(CLASS_DICT[dataset].keys(), per_class_weights)}

    result_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=["Class weight"])
    result_df.to_csv(os.path.join(output_path, f"{dataset}_class_weights.csv"), index=True)

    return per_class_weights
