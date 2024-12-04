import subprocess
import os

for dataset in ['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec', 'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']:
    if os.path.exists(os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference/', f'{dataset}')):
        continue
    args = [
        "--model", "/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/checkpoints/CellViT-256-x40.pth",
        "--dataset", f"{dataset}",
        "--outdir", "/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference",
        "--magnification", "40",
        "--data", "/mnt/lustre-grete/usr/u12649/scratch/data/test",
    ]

    command = ['python3', '/user/titus.griebel/u12649/CellViT/cell_segmentation/inference/inference_cellvit_experiment_monuseg.py'] + args

    subprocess.run(command)
