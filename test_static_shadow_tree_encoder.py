import unittest

import numpy as np

from sparse_shadow_tree import SparseShadowTreeEncoder
from static_shadow_tree_encoder import CppSSTEncoderUnavailable, CppSparseShadowTreeEncoder
from entry_point import _parse_shadow_resolution


def decode_compact(encoded):
    decoder = SparseShadowTreeEncoder(
        tile_size=encoded.tile_size,
        min_leaf_size=encoded.min_leaf_size,
        shadow_bias=float(encoded.stats.shadow_bias),
    )
    decoder._branch_10bit_start_level = int(encoded.stats.branch_10bit_start_level)
    return decoder.decode_compact(
        (encoded.stats.height, encoded.stats.width),
        encoded.compact_words,
        encoded.compact_tile_roots,
        encoded.tile_grid,
    )


class StaticShadowTreeCppEncoderTests(unittest.TestCase):
    def compare_case(self, depth, second_depth=None, tolerance=1e-6, **overrides):
        options = dict(
            tile_size=32,
            min_leaf_size=2,
            plane_error_threshold=0.0015,
            constant_epsilon=0.0005,
            use_dual_layer=second_depth is not None,
            dual_depth_slack=0.0015,
            dual_max_leak=0.0015,
            dual_visibility_tolerance=0.0015,
            shadow_bias=0.0015,
        )
        options.update(overrides)
        try:
            python_encoded = SparseShadowTreeEncoder(**options).encode(depth, second_depth)
            cpp_encoded = CppSparseShadowTreeEncoder(**options).encode(depth, second_depth)
        except CppSSTEncoderUnavailable as exc:
            self.skipTest(str(exc))

        python_decoded = decode_compact(python_encoded)
        cpp_decoded = decode_compact(cpp_encoded)
        self.assertEqual(python_encoded.stats.node_count, cpp_encoded.stats.node_count)
        self.assertTrue(cpp_encoded.stats.packed_decode_valid)
        self.assertLessEqual(float(np.max(np.abs(python_decoded - cpp_decoded))), tolerance)

    def test_constant(self):
        self.compare_case(np.full((32, 32), 0.42, dtype=np.float32))

    def test_plane(self):
        y, x = np.mgrid[0:64, 0:64]
        depth = (0.2 + 0.3 * ((x + 0.5) / 64.0) + 0.1 * ((y + 0.5) / 64.0)).astype(np.float32)
        self.compare_case(depth)

    def test_step(self):
        _y, x = np.mgrid[0:64, 0:64]
        self.compare_case(np.where(x < 32, 0.25, 0.75).astype(np.float32))

    def test_random_non_square_dual_layer(self):
        rng = np.random.default_rng(7)
        depth = rng.random((65, 67), dtype=np.float32) * 0.7 + 0.1
        second_depth = np.minimum(depth + 0.0015, 1.0).astype(np.float32)
        self.compare_case(depth, second_depth, tolerance=5e-5)

    def test_non_square_resolution_parser(self):
        self.assertEqual(_parse_shadow_resolution("7522x4311"), (7522, 4311))
        self.assertEqual(_parse_shadow_resolution("8k"), 8192)


if __name__ == "__main__":
    unittest.main()
