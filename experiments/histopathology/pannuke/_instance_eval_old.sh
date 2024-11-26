#! /bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH --job-name=eval_instance

source ~/.bashrc
conda activate sam2
python ../evaluate_ais.py -m vit_b -d pannuke -c ~/models/old_pannuke_sam/best.pt --experiment_folder /mnt/lustre-grete/usr/u12649/scratch/models/old_pannuke_sam_eval/pannuke_eval/instance