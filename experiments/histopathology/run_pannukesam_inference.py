import subprocess
import os
import shutil
print('test')

def run_inference(model_dir, input_dir, output_dir):
    for dataset in ['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lynsec', 'lizard', 'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']:
        output_path = os.path.join(output_dir, dataset, 'instance')
        os.makedirs(output_path, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, dataset, 'instance/results/instance_segmentation_with_decoder.csv')):
            continue
        args = [
            "-m", "vit_b",
            "-c", f"{model_dir}",
            "-d", f"{dataset}",
            "--experiment_folder", f"{output_path}",
            "-i", f"{input_dir}",
        ]
        command = ['python3', '/user/titus.griebel/u12649/patho-sam/experiments/histopathology/evaluate_ais.py'] + args
        print(f'Running inference with pannuke_sam model on {dataset} dataset...')
        subprocess.run(command)

        print(f'Successfully ran inference with pannuke_sam model on {dataset} dataset')


run_inference(model_dir='/mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam/checkpoints/vit_b/pannuke_sam/best.pt',
              input_dir='/mnt/lustre-grete/usr/u12649/scratch/data',
              output_dir='/mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam/inference/')
