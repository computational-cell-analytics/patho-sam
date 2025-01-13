import os
from tukra.io import read_image
from tukra.evaluation import evaluate_predictions
from tukra.inference import segment_using_instanseg
from elf.evaluation import mean_segmentation_accuracy
from skimage.measure import label
from natsort import natsorted
import pandas as pd
from glob import glob
import imageio.v3 as imageio
import numpy as np
from tqdm import tqdm

DATASETS = [
    "consep",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "lizard",
    "lynsec_he",
    "lynsec_ihc",
    "monusac",
    "monuseg",
    "nuclick",
    "nuinsseg",
    "pannuke",
    "puma",
    "srsanet",
    "tnbc",
]


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
    assert len(gt_paths) == len(
        prediction_paths
    ), f"label / prediction mismatch: {len(gt_paths)} / {len(prediction_paths)}"
    msas, sa50s, sa75s = [], [], []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths, strict=False),
        desc="Evaluate predictions",
        total=len(gt_paths),
        disable=not verbose,
    ):
        assert os.path.exists(gt_path), gt_path
        assert os.path.exists(pred_path), pred_path

        gt = imageio.imread(gt_path)
        gt = label(gt)
        pred = imageio.imread(pred_path)

        msa, scores = mean_segmentation_accuracy(pred, gt, return_accuracies=True)
        sa50, sa75 = scores[0], scores[5]
        msas.append(msa), sa50s.append(sa50), sa75s.append(sa75)

    return msas, sa50s, sa75s


def evaluate_instanseg(prediction_dir, label_dir, result_dir, dataset):
    gt_paths = natsorted(glob(os.path.join(label_dir, dataset, "loaded_testset/eval_split/test_labels/*.tiff")))
    for checkpoint in ["instanseg"]:
        save_path = os.path.join(result_dir, dataset, checkpoint, "ais_result.csv")
        if os.path.exists(save_path):
            continue
        prediction_paths = natsorted(glob(os.path.join(prediction_dir, dataset, "*.tiff")))
        if len(prediction_paths) == 0:
            print(f"No predictions for {dataset} dataset on {checkpoint} checkpoint found")
            continue
        os.makedirs(os.path.join(result_dir, dataset, checkpoint), exist_ok=True)
        print(f"Evaluating {dataset} dataset on InstanSeg ...")
        msas, sa50s, sa75s = _run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths)
        results = pd.DataFrame.from_dict(
            {
                "mSA": [np.mean(msas)],
                "SA50": [np.mean(sa50s)],
                "SA75": [np.mean(sa75s)],
            }
        )

        results.to_csv(save_path, index=False)
        print(results.head(2))

 
def infer_instanseg(data_dir, output_path):
    image_paths = natsorted(glob(os.path.join(data_dir, "test_images", "*.tiff")))
    os.makedirs(output_path, exist_ok=True)
    for image_path in image_paths:
        image = read_image(image_path)
        segmentation = segment_using_instanseg(image=image, model_type="brightfield_nuclei", target="nuclei", scale="small")
        imageio.imwrite(os.path.join(output_path, os.path.basename(image_path)), segmentation)



def run_inference(input_dir, model_dir):
    for dataset in DATASETS:
        output_path = os.path.join(model_dir, 'inference', dataset)
        input_path = os.path.join(input_dir, dataset, "loaded_testset", "eval_split")
        if os.path.exists(output_path):
            if len(os.listdir(output_path)) > 1:
                continue
        os.makedirs(output_path, exist_ok=True)
        print(f"Running inference with InstanSeg model on {dataset} dataset... \n")
        infer_instanseg(input_path, output_path)
        print(f"Inference on {dataset} dataset with the InstanSeg model successfully completed \n")
        evaluate_instanseg(
            prediction_dir=os.path.join(model_dir, 'inference'), 
            label_dir=input_dir, 
            result_dir=os.path.join(model_dir, 'results'),
            dataset=dataset
            )


def main():
    run_inference(
    input_dir="/mnt/lustre-grete/usr/u12649/data/final_test",
    model_dir="/mnt/lustre-grete/usr/u12649/models/instanseg",
    )


if __name__ == "__main__":
    main()


