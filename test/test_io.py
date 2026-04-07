import unittest


class TestReadWSI(unittest.TestCase):

    def test_file_not_found(self):
        from patho_sam.io import read_wsi
        with self.assertRaises(FileNotFoundError):
            read_wsi("/nonexistent/path/image.svs")

    @unittest.skipIf(
        __import__("importlib").util.find_spec("slideio") is None, "slideio is not installed"
    )
    def test_read_wsi_example(self):
        import os
        from micro_sam.util import get_cache_directory
        from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data
        from patho_sam.io import read_wsi

        cache_dir = os.path.join(get_cache_directory(), "sample_data")
        wsi_path = fetch_wholeslide_histopathology_example_data(cache_dir)

        # Read a small ROI crop.
        roi = (0, 0, 512, 512)
        image = read_wsi(wsi_path, image_size=roi)

        self.assertEqual(image.ndim, 3)  # H x W x C
        self.assertEqual(image.shape[0], 512)
        self.assertEqual(image.shape[1], 512)
        self.assertEqual(image.shape[2], 3)  # RGB

    @unittest.skipIf(
        __import__("importlib").util.find_spec("slideio") is None, "slideio is not installed"
    )
    def test_read_wsi_with_scale(self):
        import os
        from micro_sam.util import get_cache_directory
        from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data
        from patho_sam.io import read_wsi

        cache_dir = os.path.join(get_cache_directory(), "sample_data")
        wsi_path = fetch_wholeslide_histopathology_example_data(cache_dir)

        # Read with a target scale.
        image = read_wsi(wsi_path, scale=(256, 0))

        self.assertEqual(image.ndim, 3)
        # slideio preserves aspect ratio, so height may not be exactly 256
        self.assertLessEqual(image.shape[0], 256)


if __name__ == "__main__":
    unittest.main()
