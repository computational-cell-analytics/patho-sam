#! /bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH --job-name=eval_points

source ~/.bashrc
mamba activate sam2
python iterative_prompting_cryonuseg.py -m vit_b -c /mnt/lustre-grete/usr/u12649/scratch/models/checkpoints/vit_b/pannuke_sam/best.pt -d puma --experiment_folder /mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam_eval/puma_eval/points