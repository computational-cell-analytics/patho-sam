import os
import shutil
import subprocess
import itertools
from datetime import datetime


def submit_batch_script(script_name, decoder_only, decoder_from_pretrained, save_root, dry):
    batch_script = """#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete-h100:shared
#SBATCH -t 2-00:00:00
#SBATCH -G H100:1
#SBATCH -A gzz0001
#SBATCH --job-name=patho-sam

source ~/.bashrc
micromamba activate super
"""
    # Prepare the python scripts
    python_script = "python train_pannuke.py "
    python_script += f"-s {save_root} "

    if decoder_only:
        python_script += "--decoder_only "

    if decoder_from_pretrained:
        python_script += "--decoder_from_pretrained "

    # Add the python script to the bash script
    batch_script += python_script
    with open(script_name, "w") as f:
        f.write(batch_script)

    if not dry:
        cmd = ["sbatch", script_name]
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "patho-sam-semantic-segmentation"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def main(args):
    tmp_folder = "./gpu_jobs"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    for decoder_only, decoder_from_pretrained in itertools.product(
        [True, False], [True, False]
    ):
        submit_batch_script(
            script_name=get_batch_script_names(tmp_folder),
            decoder_only=decoder_only,
            decoder_from_pretrained=decoder_from_pretrained,
            save_root=args.save_root,
            dry=args.dry,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument(
        "-s", "--save_root", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/semantic/models"
    )
    args = parser.parse_args()
    main(args)
