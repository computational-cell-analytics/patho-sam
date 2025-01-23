import os

from micro_sam.evaluation import inference
from micro_sam.evaluation.evaluation import run_evaluation_for_iterative_prompting

from util import get_default_arguments, get_model, get_test_paths


def _run_iterative_prompting(exp_folder, predictor, start_with_box_prompt, use_masks, input_path):
    prediction_root = os.path.join(exp_folder, "start_with_box" if start_with_box_prompt else "start_with_point")
    embedding_folder = os.path.join(exp_folder, "embeddings")
    image_paths, gt_paths = get_test_paths(input_path)
    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt,
        use_masks=use_masks,
    )
    return prediction_root


def _evaluate_iterative_prompting(prediction_root, start_with_box_prompt, exp_folder, input_path):
    _, gt_paths = get_test_paths(input_path)

    run_evaluation_for_iterative_prompting(
        gt_paths=gt_paths,
        prediction_root=prediction_root,
        experiment_folder=exp_folder,
        start_with_box_prompt=start_with_box_prompt,
    )


def main():
    args = get_default_arguments()

    start_with_box_prompt = args.box  # overwrite to start first iters' prompt with box instead of single point

    # Get the predictor to perform inference
    predictor = get_model(model_type=args.model, ckpt=args.checkpoint)

    prediction_root = _run_iterative_prompting(
        args.experiment_folder, predictor, start_with_box_prompt, args.use_masks, args.input_path
    )
    _evaluate_iterative_prompting(prediction_root, start_with_box_prompt, args.experiment_folder, args.input_path)


if __name__ == "__main__":
    main()
