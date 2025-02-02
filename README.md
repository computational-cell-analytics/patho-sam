# Segment Anything for Histopathology

<a href="https://github.com/computational-cell-analytics/patho-sam"><img src="docs/logos/logo.png" width="400" align="right">

PathoSAM implements interactive annotation and (automatic) semantic segmentation for histopathology images. It is built on top of [Segment Anything](https://segment-anything.com/) by Meta AI and specializes it for histopathology data. Its core components are:
- The `patho_sam` publicly available model for interactive data annotation in 2d and 3d data that are fine-tuned on publicly available histopathology images.
- The `patho_sam` library provides training frameworks, inspired by [Segment Anything for Microscopy](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html), for downstream tasks:
    - Apply Segment Anything to histopathology images (and even whole-slide images) or fine-tune it on your data.
    - Supports downstream semantic segmentation.

Based on these components, `patho_sam` enables fast interactive and automatic annotation for histopathology images.

> NOTE: Support for running `micro-sam` on WSIs coming soon!

## Installation

How to install `patho_sam` python library from source:

We recommend to first setup an environment with the necessary requirements:

- `environment.yaml`: to set up an environment on Linux or Mac OS.
- `environment_cpu_win.yaml`: to set up an environment on Windows with CPU support.
- `environment_gpu_win.yaml`: to set up an environment on Windows with GPU support.

To create one of these environments and install `patho_sam` into it follow these steps:

1. Clone the repository: `git clone https://github.com/computational-cell-analytics/patho-sam`
2. Enter it: `cd patho-sam`
3. Create the respective environment: `conda env create -f <ENV_FILE>.yaml`
4. Activate the environment: `conda activate patho-sam`
5. Install `patho_sam`: `pip install -e .`

## Citation

If you are using this repository in your research please cite:
- [our preprint](TODO)
- and the original [Segment Anything](https://arxiv.org/abs/2304.02643) publication.
- If you use the microscopy generalist models, please also cite [Segment Anything for Microscopy](https://doi.org/10.1101/2023.08.21.554208) publication.
