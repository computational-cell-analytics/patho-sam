import os
import unittest
from shutil import rmtree

import numpy as np
from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


class TestAutomaticSegmentationWSI(unittest.TestCase):
    tmp_folder = "tmp-patho-sam-test"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.tmp_folder, exist_ok=True)
        cls.wsi_path = fetch_wholeslide_histopathology_example_data(DATA_CACHE)
        cls.roi = (0, 0, 512, 512)  # small crop for fast testing

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmp_folder, ignore_errors=True)

    def test_instance_segmentation(self):
        from patho_sam.automatic_segmentation import automatic_segmentation_wsi

        output_path = os.path.join(self.tmp_folder, "instances.tif")
        result = automatic_segmentation_wsi(
            input_image=self.wsi_path,
            model_type="vit_b_histopathology",
            roi=self.roi,
            output_path=output_path,
            tile_shape=(384, 384),
            halo=(64, 64),
            output_choice="instances",
            verbose=False,
        )

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (self.roi[3], self.roi[2]))

    def test_semantic_segmentation(self):
        from patho_sam.automatic_segmentation import automatic_segmentation_wsi

        output_path = os.path.join(self.tmp_folder, "semantic.tif")
        result = automatic_segmentation_wsi(
            input_image=self.wsi_path,
            model_type="vit_b_histopathology",
            roi=self.roi,
            output_path=output_path,
            tile_shape=(384, 384),
            halo=(64, 64),
            output_choice="semantic",
            verbose=False,
        )

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (self.roi[3], self.roi[2]))
        # Class labels should be in range [0, 5] (6 PanNuke classes incl. background)
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertLessEqual(int(result.max()), 5)

    def test_all_segmentation(self):
        from patho_sam.automatic_segmentation import automatic_segmentation_wsi

        output_path = os.path.join(self.tmp_folder, "all.tif")
        result = automatic_segmentation_wsi(
            input_image=self.wsi_path,
            model_type="vit_b_histopathology",
            roi=self.roi,
            output_path=output_path,
            tile_shape=(384, 384),
            halo=(64, 64),
            output_choice="all",
            verbose=False,
        )

        # "all" stacks instance + semantic → (2, H, W)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape, (2, self.roi[3], self.roi[2]))

    def test_instance_segmentation_idempotent(self):
        """Running twice with the same output path should load from cache."""
        from patho_sam.automatic_segmentation import automatic_segmentation_wsi

        output_path = os.path.join(self.tmp_folder, "instances_cached.tif")
        kwargs = dict(
            input_image=self.wsi_path,
            model_type="vit_b_histopathology",
            roi=self.roi,
            output_path=output_path,
            tile_shape=(384, 384),
            halo=(64, 64),
            output_choice="instances",
            verbose=False,
        )

        result1 = automatic_segmentation_wsi(**kwargs)
        result2 = automatic_segmentation_wsi(**kwargs)
        np.testing.assert_array_equal(result1, result2)


if __name__ == "__main__":
    unittest.main()
