import subprocess

for dataset in ['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec', 'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']:
    args = [
        "--model", "/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/checkpoints/CellViT-256-x40.pth",
        "--dataset", f"{dataset}",
        "--outdir", "/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference",
        "--magnification", "40",
        "--data", "/mnt/lustre-grete/usr/u12649/scratch/data",
    ]

    command = ['python3', '/user/titus.griebel/u12649/patho-sam/experiments/histopathology/evaluate_ais.py'] + args

    subprocess.run(command)

    