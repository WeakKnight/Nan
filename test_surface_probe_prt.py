import math
import unittest

import numpy as np

from surface_probe_prt import (
    evaluate_prt_l2_rgb,
    surface_probe_sh_l2_basis,
)


class SurfaceProbePrtMathTests(unittest.TestCase):
    def test_shader_basis_reference_is_orthonormal_under_sphere_sampling(self):
        sample_count = 20000
        index = np.arange(sample_count, dtype=np.float64)
        y = 1.0 - 2.0 * (index + 0.5) / sample_count
        phi = 2.0 * math.pi * np.mod(index * 0.6180339887498949, 1.0)
        radius = np.sqrt(np.maximum(1.0 - y * y, 0.0))
        directions = np.stack(
            (radius * np.cos(phi), y, radius * np.sin(phi)), axis=1
        )
        basis = np.stack(
            [surface_probe_sh_l2_basis(direction) for direction in directions]
        )
        gram = (4.0 * math.pi / sample_count) * basis.T @ basis
        np.testing.assert_allclose(gram, np.eye(9), atol=2.0e-3)

    def test_constant_environment_matches_pi_irradiance(self):
        color = np.asarray((1.25, 0.5, 2.0), dtype=np.float64)
        transport = np.zeros((9, 3), dtype=np.float64)
        # Integral of max(n dot w, 0) * Y00 over the sphere.
        transport[0] = math.pi * 0.2820947918
        lighting_sh = np.zeros((9, 3), dtype=np.float64)
        # Integral of a constant environment times Y00.
        lighting_sh[0] = color * math.sqrt(4.0 * math.pi)
        result = evaluate_prt_l2_rgb(transport, lighting_sh)
        np.testing.assert_allclose(result, math.pi * color, rtol=2.0e-9)

    def test_static_source_is_preserved_and_negative_ringing_is_clamped(self):
        transport = np.zeros((9, 3), dtype=np.float64)
        lighting_sh = np.zeros((9, 3), dtype=np.float64)
        transport[0] = (1.0, -2.0, 0.5)
        lighting_sh[0] = (2.0, 2.0, 2.0)
        result = evaluate_prt_l2_rgb(
            transport, lighting_sh, static_source=(0.25, 0.5, 0.75)
        )
        np.testing.assert_allclose(result, (2.25, 0.0, 1.75))


if __name__ == "__main__":
    unittest.main()
