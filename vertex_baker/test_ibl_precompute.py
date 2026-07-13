import os
import sys
import unittest
from pathlib import Path

import numpy as np
import slangpy as spy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ibl_precompute import EnvironmentIBL


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PATH = ROOT / "vertex_baker" / "bloem_field_sunrise_2k.hdr"
PMR_LUT_PATH = (
    ROOT.parent
    / "GDC23_PracticalMobileRendering"
    / "PracticalMobileRendering"
    / "Assets"
    / "Resources"
    / "Textures"
    / "SpecularOcclusionLut3D.exr"
)


@unittest.skipUnless(ENVIRONMENT_PATH.is_file(), "default viewer environment map is unavailable")
class IBLPrecomputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = spy.Device(
            enable_debug_layers=False,
            compiler_options={"include_paths": [ROOT, ROOT / "vertex_baker"]},
        )
        cls.environment = EnvironmentIBL(cls.device, ENVIRONMENT_PATH)
        cls.specular_occlusion = np.asarray(
            cls.environment.specular_occlusion.to_numpy(),
            dtype=np.float32,
        )

    def test_pmr_specular_occlusion_lut_is_finite_and_bounded(self):
        self.assertEqual(self.specular_occlusion.shape, (16, 256))
        self.assertTrue(np.isfinite(self.specular_occlusion).all())
        self.assertGreaterEqual(float(self.specular_occlusion.min()), 0.0)
        self.assertLessEqual(float(self.specular_occlusion.max()), 1.0)

    @unittest.skipUnless(PMR_LUT_PATH.is_file(), "PMR reference LUT repository is unavailable")
    def test_pmr_specular_occlusion_lut_matches_reference_asset(self):
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is unavailable for reading the PMR EXR reference")

        reference = cv2.imread(str(PMR_LUT_PATH), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(reference)
        reference = np.flipud(np.asarray(reference[..., 0], dtype=np.float32))
        difference = np.abs(self.specular_occlusion - reference)

        self.assertLess(float(difference.mean()), 1e-4)
        self.assertLess(float(np.percentile(difference, 99.0)), 5e-4)
        # A few samples lie directly on binary cone boundaries and differ by one QMC hit.
        self.assertLess(float(difference.max()), 0.011)


if __name__ == "__main__":
    unittest.main()
