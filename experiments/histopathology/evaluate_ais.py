import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_instance_segmentation_with_decoder


from util import get_pred_paths, get_default_arguments, VANILLA_MODELS, get_test_paths, get_val_paths


def run_instance_segmentation_with_decoder_inference(model_type, checkpoint, experiment_folder, dataset):
    val_image_paths, val_gt_paths = get_val_paths(dataset)
    test_image_paths, _ = get_test_paths(dataset)
    prediction_folder = run_instance_segmentation_with_decoder(
        checkpoint,
        model_type,
        experiment_folder,
        val_image_paths,
        val_gt_paths,
        test_image_paths
    )
    return prediction_folder


def eval_instance_segmentation_with_decoder(prediction_folder, experiment_folder, dataset):
    print("Evaluating", prediction_folder)
    _, gt_paths = get_test_paths(dataset)
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "instance_segmentation_with_decoder.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()

    if args.checkpoint is None:
        ckpt = VANILLA_MODELS[args.model]
    else:
        ckpt = args.checkpoint
    prediction_folder = run_instance_segmentation_with_decoder_inference(
        args.model, ckpt, args.experiment_folder, args.dataset
    )
    eval_instance_segmentation_with_decoder(prediction_folder, args.experiment_folder, args.dataset)


if __name__ == "__main__":
    main()
