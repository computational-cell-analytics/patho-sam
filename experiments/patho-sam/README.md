# Scripts for Evaluating Segment Anything Models for Histopathology

This folder contains scripts for evaluating interactive and automatic instance segmentation of nucleus in histopathology images.

> TLDR: The scripts expect path to input images and corresponding labels. You can write your own function to automate the processes in the scripts. The keywords mean: 1) AIS (Automatic Instance Segmentation), 2) AMG (Automatic Mask Generation), 3) Iterative Prompting (i.e. interactive segmentation, where prompts are derived from the ground-truth labels, simulating a user, and add additional points for 7 iterations). See our [preprint](https://doi.org/10.48550/arXiv.2502.00408) for more details on this.