from pdl1_detection.pdl1_convenience import infer_on_image_pdl1_positive
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", "-m", type=str,
                        default="/user/titus.griebel/u23324/ignite-data-toolkit/data/models"),
    parser.add_argument("--image_dir", "-i", type=str,
                        default="/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_pdl1_ihc/images"),
    parser.add_argument("--output_dir", "-o", type=str,
                        default="/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_pdl1_ihc/ignite_output"),
    parser.add_argument("--pred_target", type=str, choices=["nuclei", "pdl1"], required=True)
    args = parser.parse_args()

    infer_on_image_pdl1_positive(
        model_root=args.model_root,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        pred_target=args.pred_target
    )


if __name__ == "__main__":
    main()
