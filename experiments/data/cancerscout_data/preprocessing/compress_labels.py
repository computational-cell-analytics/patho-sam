import os
import imageio.v3 as imageio
from tqdm import tqdm
ROOT = "/mnt/ceph-hdd/cold/nim00020/hannibal_data"
i = 0
for dir_path, dirnames, filenames in os.walk(ROOT):
    if not dir_path.startswith("rois"):
        continue
    for filename in tqdm(filenames):
        if not filename.endswith(".tiff"):
            continue
        file_path = os.path.join(dir_path, filename)
        data = imageio.imread(file_path)
        tmp_path = file_path + ".tmp"
        imageio.imwrite(
            tmp_path,
            data,
            plugin="tifffile",
            compression="zlib"
        )
        os.replace(tmp_path, file_path)
        # print("Successfully rewrote image")
        i += 1

print(f"{i} images processed!")