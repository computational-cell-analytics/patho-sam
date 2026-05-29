import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from elf.evaluation import matching
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from micro_sam.util import precompute_image_embeddings
from tqdm import tqdm

from patho_sam.annotation import postprocess_instance_mask

ROOT = Path("/mnt/ceph-hdd/cold/nim00020/hannibal_data")
CKP_PATH = ROOT / "pathosam-models/cancerscout_instance/checkpoints/pathosam-cancerscout-instance/best.pt"


def evaluate_cancerscout_instance_segmentation(
    path: Path,
    checkpoint_path,
    intensity_threshold: int = 210,
    area_threshold: int = 120,
    device: str = "cpu",
    generalist=False,
):
    if device == "gpu":
        predictor, apg_segmenter = get_predictor_and_segmenter(
            "vit_b_histopathology", checkpoint=checkpoint_path, segmentation_mode="apg", is_tiled=True
        )
        predictor, ais_segmenter = get_predictor_and_segmenter(
            "vit_b_histopathology", checkpoint=checkpoint_path, segmentation_mode="ais", is_tiled=True
        )

        tile_shape = (384, 384)
        halo = (64, 64)

    result_dir = path / "eval_models" / "instance_segmentation_results"
    result_dir.mkdir(exist_ok=True, parents=True)

    eval_h5_paths = list(path.glob("eval_models/*data/fixed_h5_files/*.h5"))
    embedding_dir = eval_h5_paths[0].parent.parent / "cs_specialist_embeddings"
    embedding_dir.mkdir(exist_ok=True)

    for eval_h5_path in tqdm(eval_h5_paths):
        embedding_path = embedding_dir / eval_h5_path.stem
        with h5py.File(eval_h5_path, "a") as f:
            if device == "gpu":
                if generalist:
                    continue
                if "inst_preds/cs_spec_ais" in f:
                    print("Prediction already done!")
                    continue
                if embedding_path.exists():
                    shutil.rmtree(embedding_path)
                img = f["img"][:].transpose(1, 2, 0)
                precompute_image_embeddings(
                    predictor, img, save_path=embedding_path, ndim=2, tile_shape=tile_shape, halo=halo, batch_size=8
                )

                apg_pred = automatic_instance_segmentation(
                    predictor=predictor,
                    segmenter=apg_segmenter,
                    input_path=img,
                    batch_size=10,
                    optimize_memory=True,
                    embedding_path=embedding_path,
                    ndim=2,
                    tile_shape=tile_shape,
                    halo=halo,
                )

                ais_pred = automatic_instance_segmentation(
                    predictor=predictor,
                    segmenter=ais_segmenter,
                    input_path=img,
                    batch_size=10,
                    optimize_memory=True,
                    embedding_path=embedding_path,
                    ndim=2,
                    tile_shape=tile_shape,
                    halo=halo,
                )

                apg_pred_postproc = postprocess_instance_mask(
                    apg_pred, img, intensity_threshold=intensity_threshold, area_threshold=area_threshold
                )
                ais_pred_postproc = postprocess_instance_mask(
                    ais_pred, img, intensity_threshold=intensity_threshold, area_threshold=area_threshold
                )
                f.create_dataset("inst_preds/cs_spec_apg", data=apg_pred_postproc, compression="gzip")
                f.create_dataset("inst_preds/cs_spec_ais", data=ais_pred_postproc, compression="gzip")

            elif device == "cpu":
                spec = "_generalist" if generalist else ""
                csv_path = result_dir / f"{eval_h5_path.stem}{spec}.csv"

                if csv_path.exists():
                    continue

                if not generalist:
                    if "inst_preds/cs_spec_ais" not in f:
                        print(f"Missing prediction in {eval_h5_path.name}")
                        continue
                    else:
                        ais_pred_postproc = f["inst_preds/cs_spec_ais"]
                        apg_pred_postproc = f["inst_preds/cs_spec_apg"]
                        label = f["inst_labels/v_2_pproc"]
                else:
                    apg_preproc = f["inst_pred"]
                    img = f["img"][:].transpose(1, 2, 0)
                    label = f["inst_labels/v_2_pproc"]
                    apg_pred_postproc = postprocess_instance_mask(
                        apg_preproc, img, intensity_threshold=intensity_threshold, area_threshold=area_threshold
                    )
                    ais_pred_postproc = None
                results_dict = defaultdict(list)

                for seg_mode, pred in {"ais": ais_pred_postproc, "apg": apg_pred_postproc}.items():
                    if pred is None:
                        continue
                    for threshold in np.arange(0.5, 1.0, 0.05):
                        stats = matching(pred, label, threshold=threshold)
                        results_dict[f"{seg_mode}_SA"].append(stats["segmentation_accuracy"])
                        results_dict[f"{seg_mode}_F1"].append(stats["f1"])
                        results_dict[f"{seg_mode}_Precision"].append(stats["precision"])
                        results_dict[f"{seg_mode}_Recall"].append(stats["recall"])

                results_dict["Threshold"] = list(np.arange(0.5, 1.0, 0.05))
                df = pd.DataFrame(results_dict)

                df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, default=ROOT)
    parser.add_argument("--checkpoint_path", "-c", type=str, default=CKP_PATH)
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--generalist", action="store_true")
    args = parser.parse_args()
    evaluate_cancerscout_instance_segmentation(
        path=args.input_path, checkpoint_path=args.checkpoint_path, generalist=args.generalist
    )


if __name__ == "__main__":
    main()
