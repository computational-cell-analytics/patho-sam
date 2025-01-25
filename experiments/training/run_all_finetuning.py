import os
import shutil
import subprocess
from datetime import datetime


N_OBJECTS = {"vit_b": 40, "vit_l": 30, "vit_h": 25}


def write_batch_script(out_path, _name, model_type, save_root, dry):
    "Writing scripts for different patho-sam finetunings."
    batch_script = """#!/bin/bash
#SBATCH -t 14-00:00:00
#SBATCH --mem 128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH -c 32
#SBATCH --qos=14d
#SBATCH --constraint=80gb
#SBATCH --job-name=patho-sam-generalist

source ~/.bashrc
micromamba activate sam \n"""

    # python script
    python_script = f"python {_name}.py "
    python_script += f"-s {save_root} "  # save root folder
    python_script += f"-m {model_type} "  # name of the model configuration
    python_script += f"--n_objects {N_OBJECTS[model_type]} "  # choice of the number of objects

    # let's add the python script to the bash script
    batch_script += python_script

    _op = out_path[:-3] + f"_{os.path.split(_name)[-1]}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    if not dry:
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "patho-sam-finetuning"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    if args.model_type is None:
        models = list(N_OBJECTS.keys())
    else:
        models = [args.model_type]

    script_name = "train_generalist_histopathology"
    print(f"Running for {script_name}")
    for model_type in models:
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            _name=script_name,
            model_type=model_type,
            save_root=args.save_root,
            dry=args.dry,
        )


def main(args):
    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save_root", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/models"
    )
    parser.add_argument("-m", "--model_type", type=str, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
