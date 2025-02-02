import os
import subprocess
from datetime import datetime

from util import DATASETS, SAM_TYPES, MODEL_NAMES


CHECKPOINTS = {
    "vanilla_sam": SAM_TYPES,
    "generalist_sam": SAM_TYPES,
    "pannuke_sam": ["vit_b"],
    "lm_sam": ["vit_b_lm"],
    "glas_sam": ["vit_b"],
    "nuclick_sam": ["vit_b"],
}


def write_batch_script(out_path, _name, model_type, dataset, dry, model_name):
    "Writing scripts for patho-sam inference."
    batch_script = """#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -x ggpu114
#SBATCH -c 16
#SBATCH --job-name=patho-sam-inference

source ~/.bashrc
conda activate sam2 \n"""

    # python script
    python_script = f"python {_name}.py "
    python_script += f"-d {dataset} "  # dataset to infer on
    python_script += f"-m {model_type} "  # name of the model configuration
    python_script += f"-n {model_name} "  # name of the model

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

    script_name = "patho-sam-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"
    model_path = "/mnt/lustre-grete/usr/u12649/models"
    script_names = ["run_ais", "run_amg"]
    for script_name in script_names:
        print(f"Running for {script_name}")
        for dataset in DATASETS:
            for model_name in MODEL_NAMES:
                for model in CHECKPOINTS[model_name]:
                    if model_name == 'glas_sam' and dataset != 'glas':
                        continue

                    if model_name == 'nuclick_sam' and dataset != 'nuclick':
                        continue

                    if script_name != 'run_amg' and model_name == 'vanilla_sam':
                        continue

                    result = os.path.join(
                        model_path, model_name, "results", dataset,
                        f"{script_name[4:]}", f"{dataset}_{model_name}_{model}_{script_name[4:]}.csv"
                    )
                    print(result)
                    if os.path.exists(result):
                        continue

                    write_batch_script(
                        out_path=get_batch_script_names(tmp_folder),
                        _name=script_name,
                        model_type=model,
                        dataset=dataset,
                        dry=args.dry,
                        model_name=model_name,
                    )


def main(args):
    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
