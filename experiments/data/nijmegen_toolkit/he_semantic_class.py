from he_segmentation.he_convenience_script import infer_on_images


infer_on_images(
    model_root="/user/titus.griebel/u23324/ignite-data-toolkit/data/models",
    image_dir="/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_new_tumor/images",
    output_dir="/mnt/ceph-hdd/cold/nim00020/hannibal_data/train_models/rois_new_tumor/ignite_output",
)
