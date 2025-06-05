# Finetuning Segment Anything Model for Histopathlogy

This folder contains scripts for finetuning a generalist model for nuclei instance segmentation in histopathology H&E stained images. Here is a brief mention of the folders:
- `generalists`: Contains scripts for training our generalist `patho-sam` models, on a large histopathology dataset for nuclei (automatic and interactive) instance segmentation, inspired by `micro-sam`.
- `specialists`: Contains scripts for training our specialist `patho-sam` models, on PanNuke (for nuclei), GlaS (for glands), NuClick (for lymphocytes) and PUMA (for neutrophils) for task-/data-specific exploration.

> NOTE: The scripts should run out-of-the-box and download most datasets automatically. For some datasets, automatic downloads are not supported. See the respective `get_generalist_datasets.py` or `get_specialist_dataset.py` for details about their downloads!
