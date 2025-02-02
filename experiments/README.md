# Segment Anything for Histopathology *Experiment Scripts*

This directory contains all experiment scripts for building and benchmarking segment anything model for histopathology.

Here is a brief description of relevant folders:
- `benchmarking`: Contains scripts to run reference methods (eg. HoVerNet, HoVerNeXt, CellViT, InstanSeg, StarDist)
- `data`: Contains scripts for preprocessing images and corresponding labels for all experiments.
- `patho-sam`: Contains scripts for running SAM-models (eg. default SAM, `micro-sam` and `patho-sam` models for automatic and interactive instance segmentation).
- `semantic_segmentation`: Contains scripts for training and evaluation semantic segmentation using `patho-sam` generalist models.
- `training`: Contains scripts for training the specialist and generalist `patho-sam` models.

> NOTE 1: There are scripts where the expected filepaths / directories might be hard-coded. Replace them with your respective paths where you would like to store / fetch files from.

>NOTE 2: We provide [example scripts](../examples/) for convenience. Feel free to check them out for both interactive and automatic segmentation using `PathoSAM` models.
