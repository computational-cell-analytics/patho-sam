#! /bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH --job-name=eval_amg

source ~/.bashrc
mamba activate sam2
python evaluate_amg.py -m vit_b -d pannuke --experiment_folder /mnt/lustre-grete/usr/u12649/scratch/models/vanilla_sam_eval/pannuke_eval/amg