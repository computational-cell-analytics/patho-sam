import os
import shutil
from glob import glob

from natsort import natsorted


def split_pannuke(path):
    # Get the list of image file paths
    pannuke_images = natsorted(glob(os.path.join(path, "pannuke/loaded_dataset/complete_dataset/images/*.tiff")))
    print(len(pannuke_images))
    # Total number of images
    target = len(pannuke_images)

    # Define the chunk sizes
    chunk_sizes = [545, 546, 546, 545, 540]

    # Ensure the sum of chunk sizes matches the total number of images
    assert sum(chunk_sizes) == target, "Chunk sizes do not match the total number of images."

    # Split the list into chunks
    chunk1 = pannuke_images[: chunk_sizes[0]]
    chunk2 = pannuke_images[chunk_sizes[0] : sum(chunk_sizes[:2])]
    chunk3 = pannuke_images[sum(chunk_sizes[:2]) : sum(chunk_sizes[:3])]
    chunk4 = pannuke_images[sum(chunk_sizes[:3]) : sum(chunk_sizes[:4])]
    chunk5 = pannuke_images[sum(chunk_sizes[:4]) :]

    # Verify the lengths of the chunks
    assert len(chunk1) + len(chunk2) + len(chunk3) + len(chunk4) + len(chunk5) == target, "Chunk sizes are incorrect."

    print("Success")
    for chunk in [chunk5]:
        for image in chunk:
            print(image)
            shutil.copy(image, os.path.join(path, "chunk5", os.path.basename(image)))


split_pannuke("/mnt/lustre-grete/usr/u12649/scratch/data/test")
