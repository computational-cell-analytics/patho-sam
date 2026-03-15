import tarfile
import os
tar_file = '/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models.tar'

output_dir = '/mnt/ceph-hdd/cold/nim00020/hannibal_data/eval_models/CancerScout_Lung/new_tumor/extract'
with tarfile.open(tar_file) as tar:
    member = tar.getmember('home/tiga/data/transfer/CancerScout_Lung/A2020-001036_1-1-1_HE-2021-10-08T15-11-11.tiff')
    member.name = os.path.basename(member.name)
    tar.extract(member, path=output_dir)