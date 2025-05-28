# Example Scripts

Examples for using the [`micro_sam`](https://github.com/computational-cell-analytics/micro-sam) annotation tools:

- `annotator_2d_wsi.py`: Run the interactive 2d annotation tool, suitable for whole-slide images (WSIs).
- `annotator_2d.py`: Run the interactive 2d annotation tool.
- `train_pannuke_semantic.py`: Train a model for semantic segmentation of nuclei in PanNuke histopathology images.

There are Jupyter Notebooks available for using automatic segmentation and finetuning on some example data in the [notebooks](https://github.com/computational-cell-analytics/micro-sam/tree/master/notebooks) folder located in the `micro-sam` repository.

The folder `finetuning` contains example scripts that show how a Segment Anything model can be fine-tuned on custom data with the `micro_sam.train` library, and how the finetuned models can then be used within the `micro_sam` annotation tools.
