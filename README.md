# Segment Anything for Histopathology

<a href="https://github.com/computational-cell-analytics/patho-sam"><img src="docs/logos/logo.png" width="400" align="right">

PathoSAM implements interactive annotation and (automatic) instance and semantic segmentation for histopathology images. It is built on top of [Segment Anything](https://segment-anything.com/) by Meta AI and our prior work [Segment Anything for Microscopy](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html). It specializes Segment Anything for nucleus segmentation in histopathology data. Its core components are:
- The publicly available `patho_sam` models for interactive data annotation that were fine-tuned on openly available histopathology images.
- The `patho_sam` library, which provides training functionality based on [Segment Anything for Microscopy](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html), and supports:
    - Application of Segment Anything to histopathology images, including whole-slide images, and fine-tuning on your data.
    - Semantic instance segmentation.

Based on these components, `patho_sam` enables fast interactive and automatic annotation for histopathology images, see [Usage](#usage) for details.

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

## Usage

Coming soon.

## Citation

If you are using this repository in your research please cite:
- [Our preprint](TODO).
- The [Segment Anything for Microscopy](https://doi.org/10.1101/2023.08.21.554208) publication
- And the original [Segment Anything](https://arxiv.org/abs/2304.02643) publication.
