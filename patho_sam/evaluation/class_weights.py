import os
from glob import glob
from typing import List, Union, Optional

import numpy as np
import pandas as pd

from torch_em.util.image import load_data
from torch_em.data.datasets import histopathology

from ..training.util import CLASS_DICT


LABEL_KEYS = {
    'pannuke': {'semantic': 'labels/semantic', 'instance': 'labels/instances'},
    'puma': {'semantic': 'labels/semantic/nuclei', 'instance': 'labels/instance/nuclei'},
}


def extract_class_weights(
    path: Union[os.PathLike, str], dataset: Optional[str] = None, output_path: Optional[Union[os.PathLike, str]] = None,
) -> List:
    """Extract class weights per semantic class.

    Args:
        path: The filepath where the input data is either located or will be downloaded automatically.
        dataset: The choice of dataset to extract the class weights.
        output_path: The output directory where you would like to store the class weights.

    Return:
        List of class weights for chosen dataset.
    """

    # Get the input filepaths
    _get_dataset = {
        "puma": lambda: histopathology.puma.get_puma_paths(path=path, split="train", download=True),
        "pannuke": lambda: histopathology.pannuke.get_pannuke_paths(
            path=path, folds=["fold_1", "fold_2"], download=True,
        )
    }

    # Next, get the volume paths.
    volume_paths = _get_dataset[dataset]()

    # Load the entire instance and semantic stack.
    if isinstance(volume_paths, str):
        instances = load_data(path, key=LABEL_KEYS[dataset]['instance'])
        semantics = load_data(path, key=LABEL_KEYS[dataset]['semantic'])
    else:
        all_semantic, all_instances = [], []
        for fpath in glob(os.path.join(path, "*.h5")):
            _instance = load_data(fpath, key=LABEL_KEYS[dataset]['instance'])
            _semantic = load_data(fpath, key=LABEL_KEYS[dataset]['semantic'])

            # Get all semantic and instance labels.
            all_instances.append(_instance)
            all_semantic.append(_semantic)

        instances = np.concatenate(all_instances, axis=0)
        semantics = np.concatenate(all_semantic, axis=0)

    # We need the following:
    # - Count the total number of instances.
    total_instance_counts = [
        len(np.unique(ilabel)[1:]) for ilabel in instances if len(np.unique(ilabel)) > 1
    ]  # Counting all valid foreground instances only.
    total_instance_counts = sum(total_instance_counts)

    # - Count per-semantic-class instances.
    class_ids = CLASS_DICT[dataset].values()
    total_per_class_instance_counts = [
        [len(np.unique(np.where(slabel == cid, ilabel, 0))[1:]) for cid in class_ids]
        for ilabel, slabel in zip(instances, semantics) if len(np.unique(ilabel)) > 1
    ]
    assert total_instance_counts == sum([sum(t) for t in total_per_class_instance_counts])

    # Calculate per class mean values.
    total_per_class_instance_counts = [sum(x) for x in zip(*total_per_class_instance_counts)]
    assert total_instance_counts == sum(total_per_class_instance_counts)

    # Finally, let's get the weight per class. Results are saved as .csv in the output folder and output as a list
    per_class_weights = [t / total_instance_counts for t in total_per_class_instance_counts]

    # Store the class weights locally.
    os.makedirs(output_path, exist_ok=True)
    result_dict = {nt: weight for nt, weight in zip(CLASS_DICT[dataset].keys(), per_class_weights)}
    result_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=["Class Weight"])
    result_df.to_csv(os.path.join(output_path, f"{dataset}_class_weights.csv"), index=True)

    return per_class_weights
