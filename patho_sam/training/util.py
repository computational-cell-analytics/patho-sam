from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb


def histopathology_identity(x, ensure_rgb=True):
    """Identity transform.
    Inspired from 'micro_sam/training/util.py' -> 'identity' function.

    This ensures to skip data normalization when finetuning SAM.
    Data normalization is performed within the model to SA-1B data statistics
    and should thus be skipped as a preprocessing step in training.
    """
    if ensure_rgb:
        x = to_rgb(x)

    return x
