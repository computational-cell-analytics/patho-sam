import os
import subprocess
import sys
import unittest
from shutil import rmtree

from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def _run(args):
    return subprocess.run([sys.executable, "-m"] + args, capture_output=True, text=True)


class TestCLIRun(unittest.TestCase):
    """End-to-end CLI run on a small WSI crop."""

    tmp_folder = "tmp-patho-sam-cli-test"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.tmp_folder, exist_ok=True)
        cls.wsi_path = fetch_wholeslide_histopathology_example_data(DATA_CACHE)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmp_folder, ignore_errors=True)

    def test_automatic_segmentation_instances(self):
        output_path = os.path.join(self.tmp_folder, "cli_instances.tif")
        result = _run([
            "patho_sam.automatic_segmentation",
            "-i", self.wsi_path,
            "-o", output_path,
            "--roi", "0", "0", "512", "512",
            "-m", "vit_b_histopathology",
            "--output_choice", "instances",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # The CLI writes <stem>_ROI_..._instances.tif
        written = [f for f in os.listdir(self.tmp_folder) if "instances" in f]
        self.assertTrue(len(written) > 0, "No instances output file found.")

    def test_automatic_segmentation_semantic(self):
        output_path = os.path.join(self.tmp_folder, "cli_semantic.tif")
        result = _run([
            "patho_sam.automatic_segmentation",
            "-i", self.wsi_path,
            "-o", output_path,
            "--roi", "0", "0", "512", "512",
            "-m", "vit_b_histopathology",
            "--output_choice", "semantic",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        written = [f for f in os.listdir(self.tmp_folder) if "semantic" in f]
        self.assertTrue(len(written) > 0, "No semantic output file found.")


if __name__ == "__main__":
    unittest.main()
