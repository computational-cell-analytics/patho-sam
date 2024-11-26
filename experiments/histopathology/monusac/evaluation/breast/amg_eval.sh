#! /bin/bash
#SBATCH -p gpu
#SBATCH -G V100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH --job-name=eval_instance

source ~/.bashrc
mamba activate sam2
python evaluate_amg_monusac.py -m vit_b -c /scratch/users/u11644/models/checkpoints/vit_b/pannuke_sam/best.pt --experiment_folder /scratch/users/u11644/models/evaluation/monusac_eval/breast/amg_eval --organ_type breast