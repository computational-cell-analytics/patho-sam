import subprocess

args = [
    "--model", "/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/checkpoints/CellViT-256-x40.pth",
    "--dataset", "monuseg",
    "--outdir", "/mnt/lustre-grete/usr/u12649/scratch/models/cellvit/inference",
    "--magnification", "40",
    "--data", "/mnt/lustre-grete/usr/u12649/scratch/data",
]

command = ['python3', '/user/titus.griebel/u12649/CellViT/cell_segmentation/inference/inference_cellvit_experiment_monuseg.py'] + args

subprocess.run(command)
