#! /bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH --job-name=eval_points

source ~/.bashrc
mamba activate sam2
python iterative_prompting.py -m vit_b -d lynsec --experiment_folder /mnt/lustre-grete/usr/u12649/scratch/models/vanilla_sam_eval/lynsec_eval/points