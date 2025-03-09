# Segment Anything for Histopathology

<a href="https://github.com/computational-cell-analytics/patho-sam"><img src="docs/logos/logo.png" width="400" align="right"></a>

PathoSAM implements interactive annotation and (automatic) instance and semantic segmentation for histopathology images. It is built on top of [Segment Anything](https://segment-anything.com/) by Meta AI and our prior work [Segment Anything for Microscopy](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html). It specializes Segment Anything for nucleus segmentation in histopathology data. Its core components are:
- The publicly available `patho_sam` models for interactive data annotation that were fine-tuned on openly available histopathology images.
- The `patho_sam` library, which provides training functionality based on [Segment Anything for Microscopy](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html), and supports:
    - Application of Segment Anything to histopathology images, including whole-slide images, and fine-tuning on your data.
    - Semantic segmentation.

Based on these components, `patho_sam` enables fast interactive and automatic annotation for histopathology images, see [Usage](#usage) for details.

## Installation

How to install `patho_sam` python library from source:

To create one of these environments and install `patho_sam` into it follow these steps:

1. Clone the repository: `git clone https://github.com/computational-cell-analytics/patho-sam`
2. Enter it: `cd patho-sam`
3. Create the environment with the necessary requirements: `conda env create -f environment.yaml`
4. Activate the environment: `conda activate patho-sam`
5. Install `patho_sam`: `pip install -e .`

## Usage

### Using example scripts:

See the [examples](./examples/) folder for more details.

### Using CLI:

- Download the example whole-slide image by running the following via terminal: `patho_sam.example_data` (see `patho_sam.example_data -h` for more details about the CLI).
- Run automatic segmentation on your own WSI or the example data by running the following via terminal:
    ```bash
    patho_sam.automatic_segmentation -i /home/anwai/.cache/micro_sam/sample_data/whole-slide-histopathology-example-image.svs -o segmentation.tif
    ```

    > NOTE 1: See `patho_sam.automatic_segmentation -h` for more details about the CLI.

    > NOTE 2: You can find your cache directory using: `python -c "from micro_sam.util import get_cache_directory; print(get_cache_directory())"`.


## Citation

If you are using this repository in your research please cite:
- [Our preprint](https://doi.org/10.48550/arXiv.2502.00408).
- the [Segment Anything for Microscopy](https://www.nature.com/articles/s41592-024-02580-4) publication.
- And the original [Segment Anything](https://arxiv.org/abs/2304.02643) publication.
