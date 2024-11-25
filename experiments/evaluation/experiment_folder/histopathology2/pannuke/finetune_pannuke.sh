#! /bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH --job-name=pannuke_finetuning

source ~/.bashrc
mamba activate sam2
python pannuke_finetuning.py -i /mnt/lustre-grete/usr/u12649/scratch/data/pannuke -s /mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam/n_objects_5 --iterations 100000 -e /mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam --n_objects 5