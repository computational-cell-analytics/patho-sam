import os
from glob import glob
from natsort import natsorted
from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_amg
#from evaluate_instance_segmentation_monusac import get_test_paths, get_val_paths

from util_2 import get_default_arguments, get_pred_paths, VANILLA_MODELS


def get_val_paths(dataset):
    path = os.path.join('/mnt/lustre-grete/usr/u12649/scratch/data/', f'{dataset}', 'loaded_dataset/complete_dataset/test2')
    val_image_paths = natsorted(glob(os.path.join(path, 'val_images/*')))
    val_label_paths = natsorted(glob(os.path.join(path,'val_labels/*')))
    print(len(val_image_paths), len(val_label_paths))

    return val_image_paths, val_label_paths

def get_test_paths(dataset):
    path = os.path.join('/mnt/lustre-grete/usr/u12649/scratch/data/', f'{dataset}', 'loaded_dataset/complete_dataset/test2')
    test_image_paths = natsorted(glob(os.path.join(path, 'test_images/*')))
    test_label_paths = natsorted(glob(os.path.join(path, 'test_labels/*')))
    print(len(test_image_paths), len(test_label_paths))
    return test_image_paths, test_label_paths

def run_amg_inference(model_type, checkpoint, experiment_folder, dataset):
    val_image_paths, val_gt_paths = get_val_paths(dataset)
    test_image_paths, _ = get_test_paths(dataset)
    prediction_folder = run_amg(
        checkpoint,
        model_type,
        experiment_folder,
        val_image_paths,
        val_gt_paths,
        test_image_paths
    )
    return prediction_folder

def eval_amg(prediction_folder, experiment_folder, dataset):
    print("Evaluating", prediction_folder)
    _, gt_paths = get_test_paths(dataset)
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "amg.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()
    if args.checkpoint is None:
        ckpt = VANILLA_MODELS[args.model]
    else:
        ckpt = args.checkpoint

    prediction_folder = run_amg_inference(args.model, ckpt, args.experiment_folder, args.dataset)
    eval_amg(prediction_folder, args.experiment_folder, args.dataset) #deleted args.dataset as an argument for eval_amg due to error occurence


if __name__ == "__main__":
    main()
