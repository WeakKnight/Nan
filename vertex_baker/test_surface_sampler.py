import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Material, Mesh, Model
from surface_sampler import sample_model_surface


def _model(meshes):
    all_positions = np.concatenate([mesh.positions for mesh in meshes], axis=0)
    return Model(
        meshes=meshes,
        materials=[Material("default", np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32))],
        bounds_min=all_positions.min(axis=0).astype(np.float32),
        bounds_max=all_positions.max(axis=0).astype(np.float32),
    )


def _mesh(positions, indices):
    positions = np.asarray(positions, dtype=np.float32)
    return Mesh(
        name="mesh",
        positions=positions,
        normals=np.zeros_like(positions),
        uvs=np.zeros((positions.shape[0], 2), dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        material_index=0,
    )


class VertexBakerSurfaceSamplerTests(unittest.TestCase):
    def test_deterministic_for_fixed_seed(self):
        model = _model([
            _mesh(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 1, 2]],
            )
        ])
        a = sample_model_surface(model, 128, seed=5)
        b = sample_model_surface(model, 128, seed=5)
        np.testing.assert_array_equal(a.positions, b.positions)

    def test_samples_lie_inside_triangle(self):
        model = _model([
            _mesh(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 1, 2]],
            )
        ])
        samples = sample_model_surface(model, 1024, seed=6)
        self.assertTrue(np.all(samples.positions[:, 0] >= -1e-6))
        self.assertTrue(np.all(samples.positions[:, 1] >= -1e-6))
        self.assertTrue(np.all(samples.positions[:, 0] + samples.positions[:, 1] <= 1.0 + 1e-6))

    def test_area_weighted_sampling(self):
        model = _model([
            _mesh(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [10, 0, 0],
                    [14, 0, 0],
                    [10, 4, 0],
                ],
                [[0, 1, 2], [3, 4, 5]],
            )
        ])
        samples = sample_model_surface(model, 4096, seed=7)
        small_count = int(np.count_nonzero(samples.positions[:, 0] < 5.0))
        large_count = int(np.count_nonzero(samples.positions[:, 0] > 5.0))
        self.assertGreater(large_count, small_count * 8)

    def test_degenerate_triangles_are_skipped(self):
        model = _model([
            _mesh(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0],
                    [2, 1, 0],
                ],
                [[0, 1, 2], [3, 4, 5]],
            )
        ])
        samples = sample_model_surface(model, 256, seed=8)
        self.assertTrue(np.all(samples.positions[:, 0] >= 2.0 - 1e-6))


if __name__ == "__main__":
    unittest.main()
