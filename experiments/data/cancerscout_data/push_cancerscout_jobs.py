import os
import subprocess
import argparse
from glob import glob
from datetime import datetime
from natsort import natsorted
import shutil


def write_batch_script(out_path, job_name, input_path, dry, output_path, embedding_path, memory, roi):
    "Writing scripts for patho-sam inference."
    batch_script = f"""#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00020
#SBATCH -c 16
#SBATCH --mem={memory}G
#SBATCH --job-name=patho-sam-inference-{job_name}

source ~/.bashrc
mamba activate sam \n"""

    # python script
    python_script = "python automatic_segmentation.py "
    python_script += f"-i {input_path} "  # dataset to infer on
    python_script += "-m vit_l_histopathology "  # name of the model configuration
    python_script += f"-e {embedding_path} "  # name of the model configuration
    python_script += f"-o {output_path} "  # name of the model configuration
    # python_script += f"--roi {roi[0]} {roi[1]} {roi[2]} {roi[3]} "  # name of the model configuration
    # breakpoint()
    # let's add the python script to the bash script
    batch_script += python_script

    _op = out_path[:-3] + f"_{job_name}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    if not dry:
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "patho-sam-cancer-scout-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def get_required_memory(wsi_path) -> int:
    file_size_gb = os.path.getsize(wsi_path) / 1e9
    if file_size_gb < 0.5:
        return 80
    elif 0.5 <= file_size_gb < 1.2:
        return 64
    else:
        return None


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    image_type = "new_tumor_rois_2"
    # name_list = 
    name_list = ['A2020-001001_1-1-1_HE-2021-05-25T19-13-28roi_1.tiff', 'A2020-001334_1-1-1_HE-2021-10-08T22-41-22roi_0.tiff', 'A2020-001001_1-1-1_HE-2021-05-25T19-13-28roi_0.tiff', 'A2020-001191_1-1-1_HE-2021-09-09T13-54-50roi_2.tiff']
    tmp_folder = "./gpu_jobs"
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)
    input_dir = f"/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/CancerScout_Lung/{image_type}"
    output_dir = os.path.join(os.path.split(input_dir)[0], f"{image_type}_preds")
    embedding_dir = os.path.join(os.path.split(input_dir)[0], f"{image_type}_embeddings")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    for idx, wsi_path in enumerate(natsorted(glob(os.path.join(input_dir, "*.tiff")))):
        if os.path.basename(wsi_path) not in name_list:
            continue

        wsi_name = os.path.basename(wsi_path).split("_")[0]
        # rois = [(10000, 10000, 10000, 10000), (20000, 10000, 10000, 10000), (10000, 20000, 10000, 10000)]
        # for roi_idx, roi in enumerate(rois):
        roi_idx = os.path.basename(wsi_path).split("_")[-1][0]
        output_path = os.path.join(output_dir, f"{wsi_name}_pred_roi_{roi_idx}.tiff")
        if os.path.exists(output_path):
            continue
        embedding_path = os.path.join(embedding_dir, f"{wsi_name}_roi_{roi_idx}")
        os.makedirs(embedding_path, exist_ok=True)
        memory = get_required_memory(wsi_path)

        if memory is None:
            continue
        # breakpoint()
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            job_name=wsi_name,
            dry=args.dry,
            input_path=wsi_path,
            output_path=output_path,
            embedding_path=embedding_path,
            memory=memory,
            roi=None,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    submit_slurm(args)


if __name__ == "__main__":
    main()
