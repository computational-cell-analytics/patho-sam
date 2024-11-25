#! /bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH --job-name=eval_instance

source ~/.bashrc
conda activate sam2
python evaluate_instance_segmentation.py -m vit_b -d tnbc --experiment_folder /mnt/lustre-grete/usr/u12649/scratch/models/vanilla_sam_eval/tnbc_eval/instance