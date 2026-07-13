import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vertex_baking_utils import bake_least_squares, build_native
from model import load_gltf_model
from surface_sampler import sample_model_surface
from vertex_color_baker import _sample_material_nearest, _vertex_anchor_samples, bake_model_vertex_colors, sample_base_color_values


class VertexBakingUtilsTests(unittest.TestCase):
    def test_lantern_loads_core_gltf_pbr_material(self):
        root = Path(__file__).resolve().parents[1]
        model = load_gltf_model(root / "vertex_baker" / "glTF" / "Lantern.gltf")
        material = model.materials[0]

        np.testing.assert_array_equal(material.base_color, np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(material.emissive, np.ones(3, dtype=np.float32))
        self.assertEqual(material.roughness, 1.0)
        self.assertEqual(material.metallic, 1.0)
        self.assertIsNotNone(material.base_color_texture)
        self.assertIsNotNone(material.metallic_roughness_texture)
        self.assertIsNotNone(material.normal_texture)
        self.assertIsNotNone(material.emissive_texture)
        self.assertTrue(material.base_color_texture_path.is_file())
        self.assertTrue(material.metallic_roughness_texture_path.is_file())
        self.assertTrue(material.normal_texture_path.is_file())
        self.assertTrue(material.emissive_texture_path.is_file())

        metallic_roughness = material.metallic_roughness_texture
        self.assertGreater(float(np.ptp(metallic_roughness[:, :, 1])), 0.5)
        self.assertGreater(float(np.ptp(metallic_roughness[:, :, 2])), 0.5)

    @classmethod
    def setUpClass(cls):
        build_native()

    def test_single_triangle_exact_reconstruction(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        sample_triangles = np.array([0, 0, 0], dtype=np.uint32)
        bary = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        values = np.array([[0.1, 0.2, 0.3], [0.8, 0.1, 0.4], [0.2, 0.9, 0.7]], dtype=np.float32)

        baked = bake_least_squares(positions, indices, sample_triangles, bary, values)

        np.testing.assert_allclose(baked, values, atol=1e-5)

    def test_constant_value_stays_constant(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        sample_triangles = np.array([0, 0, 0, 0], dtype=np.uint32)
        bary = np.array(
            [
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ],
            dtype=np.float32,
        )
        values = np.full((4, 1), 0.42, dtype=np.float32)

        baked = bake_least_squares(positions, indices, sample_triangles, bary, values, regularization_weight=0.1)

        np.testing.assert_allclose(baked, 0.42, atol=1e-4)

    def test_lantern_vertex_color_preview_command_outputs_image(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "vertex_baker" / "out" / "test_lantern_vertex_color.png"
        if output.exists():
            output.unlink()

        subprocess.run(
            [
                sys.executable,
                str(root / "vertex_baker" / "main.py"),
                "--mode",
                "vertex-color",
                "--sample-count",
                "500",
                "--width",
                "320",
                "--height",
                "240",
                "--output",
                str(output),
            ],
            cwd=root,
            check=True,
        )

        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 0)

    def test_lantern_vertex_colors_stay_within_sampled_texture_range(self):
        root = Path(__file__).resolve().parents[1]
        model = load_gltf_model(root / "vertex_baker" / "glTF" / "Lantern.gltf")
        samples = sample_model_surface(model, 1024, seed=11)
        values = sample_base_color_values(model, samples)

        baked = bake_model_vertex_colors(model, samples, values)

        for mesh_index, mesh_values in enumerate(baked):
            mask = samples.mesh_indices == mesh_index
            if not np.any(mask):
                continue
            _, _, _, anchor_values = _vertex_anchor_samples(model, mesh_index)
            reference_values = np.concatenate([values[mask], anchor_values], axis=0)
            lower = reference_values.min(axis=0)
            upper = reference_values.max(axis=0)
            self.assertTrue(np.all(mesh_values >= lower - 1e-6))
            self.assertTrue(np.all(mesh_values <= upper + 1e-6))

    def test_batched_texture_sampling_matches_scalar_sampling(self):
        root = Path(__file__).resolve().parents[1]
        model = load_gltf_model(root / "vertex_baker" / "glTF" / "Lantern.gltf")
        samples = sample_model_surface(model, 2048, seed=13)

        batched_values = sample_base_color_values(model, samples)
        scalar_values = np.zeros_like(batched_values)
        for sample_index in range(samples.positions.shape[0]):
            mesh_index = int(samples.mesh_indices[sample_index])
            triangle_index = int(samples.triangle_indices[sample_index])
            mesh = model.meshes[mesh_index]
            material = model.materials[mesh.material_index]
            triangle = mesh.indices[triangle_index].astype(np.int64)
            bary = samples.barycentrics[sample_index]
            uv = (
                bary[0] * mesh.uvs[triangle[0]]
                + bary[1] * mesh.uvs[triangle[1]]
                + bary[2] * mesh.uvs[triangle[2]]
            )
            scalar_values[sample_index] = _sample_material_nearest(material, uv)

        np.testing.assert_array_equal(batched_values, scalar_values)

    def test_lantern_vertex_colors_have_bounded_vertex_anchor_error(self):
        root = Path(__file__).resolve().parents[1]
        model = load_gltf_model(root / "vertex_baker" / "glTF" / "Lantern.gltf")
        samples = sample_model_surface(model, 1024, seed=12)
        values = sample_base_color_values(model, samples)

        baked = bake_model_vertex_colors(model, samples, values, vertex_anchor_max_error=0.08)

        for mesh_index, mesh_values in enumerate(baked):
            anchor_vertices, _, _, anchor_values = _vertex_anchor_samples(model, mesh_index)
            if anchor_vertices.shape[0] == 0:
                continue
            error = np.abs(mesh_values[anchor_vertices] - anchor_values)
            self.assertLessEqual(float(error.max()), 0.08001)


if __name__ == "__main__":
    unittest.main()
