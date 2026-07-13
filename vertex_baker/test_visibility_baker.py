import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Material, Mesh, Model
from surface_sampler import SurfaceSamples, sample_model_surface
import pmr_visibility_reference as pmr
from vertex_baking_utils import (
    bake_least_squares,
    bake_pmr_visibility_sh_least_squares,
    bake_visibility_least_squares,
    build_native,
    pmr_sh_to_cones_native,
    trace_pmr_visibility_sh_tinybvh,
)
from tinybvh_visibility_baker import flatten_model_geometry
from interactive_viewer import export_visibility_cone_viewer
from slang_viewer import _vertex_cone_float4, _vertex_value_float4
from visibility_cone_visualizer import build_visibility_cone_line_segments
from visibility_baker import (
    HALF_PI,
    VisibilitySampleCones,
    bake_least_squares_python,
    bake_visibility_cones_python,
    compute_mesh_tangents,
    decode_visibility_cone_texcoord2,
    encode_visibility_cone_texcoord2,
    fit_visibility_cones,
    sample_visibility_cones_python,
    vertex_visibility_preview_values,
)


def _model(meshes):
    all_positions = np.concatenate([mesh.positions for mesh in meshes], axis=0)
    return Model(
        meshes=meshes,
        materials=[Material("default", np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32))],
        bounds_min=all_positions.min(axis=0).astype(np.float32),
        bounds_max=all_positions.max(axis=0).astype(np.float32),
    )


def _mesh(positions, indices, normals=None, uvs=None):
    positions = np.asarray(positions, dtype=np.float32)
    if normals is None:
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (positions.shape[0], 1))
    if uvs is None:
        uvs = positions[:, :2].astype(np.float32)
    return Mesh(
        name="mesh",
        positions=positions,
        normals=np.asarray(normals, dtype=np.float32),
        uvs=np.asarray(uvs, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        material_index=0,
    )


def _single_sample_on_triangle(mesh_index=0, triangle_index=0, bary=None):
    if bary is None:
        bary = np.array([[0.25, 0.25, 0.5]], dtype=np.float32)
    return SurfaceSamples(
        positions=np.zeros((bary.shape[0], 3), dtype=np.float32),
        barycentrics=bary.astype(np.float32),
        mesh_indices=np.full((bary.shape[0],), mesh_index, dtype=np.int32),
        triangle_indices=np.full((bary.shape[0],), triangle_index, dtype=np.int32),
    )


def _with_sample_positions(model, samples):
    positions = np.zeros_like(samples.positions)
    for sample_index in range(samples.positions.shape[0]):
        mesh = model.meshes[int(samples.mesh_indices[sample_index])]
        tri = mesh.indices[int(samples.triangle_indices[sample_index])].astype(np.int64)
        bary = samples.barycentrics[sample_index]
        positions[sample_index] = (
            bary[0] * mesh.positions[tri[0]]
            + bary[1] * mesh.positions[tri[1]]
            + bary[2] * mesh.positions[tri[2]]
        )
    return SurfaceSamples(
        positions=positions.astype(np.float32),
        barycentrics=samples.barycentrics,
        mesh_indices=samples.mesh_indices,
        triangle_indices=samples.triangle_indices,
    )


class VisibilityBakerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_native()

    def test_texcoord2_encode_decode_round_trip_matches_shader_math(self):
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        tangents = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]], dtype=np.float32)
        directions = np.array([[0.35, 0.2, 0.915], [0.2, 0.92, -0.25]], dtype=np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        apertures = np.array([0.8, 1.2], dtype=np.float32)
        scales = np.array([0.7, 0.45], dtype=np.float32)

        encoded = encode_visibility_cone_texcoord2(directions, apertures, scales, normals, tangents)
        decoded = decode_visibility_cone_texcoord2(encoded, normals, tangents)

        np.testing.assert_allclose(decoded.directions, directions, atol=2e-6)
        np.testing.assert_allclose(decoded.aperture_radians, apertures, atol=1e-6)
        np.testing.assert_allclose(decoded.scale, scales, atol=1e-6)

    def test_visibility_preview_matches_pmr_ambient_occlusion(self):
        mesh = _mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
        )
        aperture_normalized = np.array([0.25, 0.5, 0.8], dtype=np.float32)
        scale = np.array([0.9, 1.2, 0.6], dtype=np.float32)
        directions = np.array(
            [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.8, 0.0, 0.6]],
            dtype=np.float32,
        )
        cones = np.column_stack((directions, aperture_normalized * HALF_PI, scale)).astype(np.float32)
        encoded = encode_visibility_cone_texcoord2(
            directions,
            cones[:, 3],
            scale,
            mesh.normals,
            compute_mesh_tangents(mesh),
        )
        result = type("VisibilityResult", (), {"vertex_cones": [cones], "encoded_texcoord2": [encoded]})()

        actual = vertex_visibility_preview_values(result, _model([mesh]))[0][:, 0]
        cos_theta = directions[:, 2]
        corrected_cos_theta = (
            cos_theta * (1.0 - aperture_normalized)
            + (cos_theta * 0.5 + 0.5) * aperture_normalized
        )
        expected = corrected_cos_theta * aperture_normalized * np.clip(scale, 0.0, 1.0)
        np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_surface_sampling_can_enforce_minimum_samples_per_mesh(self):
        large_mesh = _mesh(
            [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
            [[0, 1, 2]],
        )
        small_mesh = _mesh(
            [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.0]],
            [[0, 1, 2]],
        )
        samples = sample_model_surface(_model([large_mesh, small_mesh]), 12, seed=4, min_samples_per_mesh=5)

        counts = np.bincount(samples.mesh_indices, minlength=2)
        self.assertGreaterEqual(int(counts[0]), 5)
        self.assertGreaterEqual(int(counts[1]), 5)
        self.assertEqual(int(samples.positions.shape[0]), 12)

    def test_pmr_surface_sampling_uses_fixed_count_per_triangle(self):
        mesh = _mesh(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
            ],
            [[0, 1, 2], [3, 4, 5]],
        )

        sampling = pmr.sample_model_surface_pmr(_model([mesh]), samples_per_triangle=4)

        counts = np.bincount(sampling.samples.triangle_indices, minlength=2)
        np.testing.assert_array_equal(counts, np.array([4, 4]))
        np.testing.assert_allclose(sampling.samples.barycentrics[:4].sum(axis=0), 4.0 / 3.0, atol=2e-7)
        self.assertEqual(sampling.samples.positions.shape[0], 8)
        self.assertEqual(
            sampling.samples.positions.shape[0],
            sampling.proxy.triangles.shape[0] * sampling.samples_per_triangle,
        )

    def test_pmr_proxy_merges_only_matching_position_and_normal(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        mesh_a = _mesh(positions, [[0, 1, 2]])
        mesh_b = _mesh(
            positions,
            [[0, 1, 2]],
            normals=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (3, 1)),
        )
        model = _model([mesh_a, mesh_b])

        normal_aware = pmr.build_pmr_mesh_proxy(model, compare_normals=True)
        position_only = pmr.build_pmr_mesh_proxy(model, compare_normals=False)

        self.assertEqual(normal_aware.positions.shape[0], 6)
        self.assertEqual(position_only.positions.shape[0], 3)

    def test_pmr_proxy_hashes_local_keys_but_keeps_world_representatives(self):
        local_positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        mesh_a = _mesh(local_positions, [[0, 1, 2]])
        mesh_b = _mesh(local_positions + np.array([5.0, 0.0, 0.0], dtype=np.float32), [[0, 1, 2]])
        mesh_a.proxy_hash_positions = local_positions.copy()
        mesh_b.proxy_hash_positions = local_positions.copy()
        mesh_a.proxy_hash_normals = mesh_a.normals.copy()
        mesh_b.proxy_hash_normals = mesh_b.normals.copy()

        proxy = pmr.build_pmr_mesh_proxy(_model([mesh_a, mesh_b]))

        self.assertEqual(proxy.positions.shape[0], 3)
        np.testing.assert_array_equal(proxy.mesh_vertex_remapping[0], proxy.mesh_vertex_remapping[1])
        np.testing.assert_allclose(proxy.positions, mesh_b.positions, atol=0.0)

    def test_pmr_full_visibility_projects_to_constant_sh(self):
        directions = pmr.make_pmr_ray_directions(512)
        sh = pmr.project_pmr_visibility_sh(np.ones((1, 512), dtype=np.float32), directions)[0]

        self.assertAlmostEqual(float(sh[0]), float(np.sqrt(4.0 * np.pi)), places=6)
        self.assertLess(float(np.max(np.abs(sh[1:]))), 0.015)

    def test_tinybvh_visibility_sh_matches_python_reference(self):
        mesh = _mesh(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [[0, 1, 2], [2, 1, 3]],
        )
        model = _model([mesh])
        sampling = pmr.sample_model_surface_pmr(model, samples_per_triangle=4)
        expected = pmr.trace_pmr_visibility_sh_python(
            model,
            sampling,
            ray_count=256,
            ray_length=0.5,
            self_bias=0.001,
        )
        positions, indices = flatten_model_geometry(model)
        actual, stats = trace_pmr_visibility_sh_tinybvh(
            positions,
            indices,
            sampling.samples.positions,
            sampling.sample_normals,
            ray_count=256,
            max_distance=0.5,
            self_bias=0.001,
            layout="auto",
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        self.assertEqual(stats.layout, "bvh")
        self.assertEqual(stats.total_ray_count, 8 * 256)
        self.assertGreater(stats.visible_ray_count, 0)
        self.assertGreater(stats.trace_milliseconds, 0.0)

    def test_tinybvh_zero_normal_surface_sample_does_not_self_occlude(self):
        positions = np.array(
            [[0.0, 20.0, 0.0], [1.0, 20.0, 0.0], [0.0, 20.0, 1.0]],
            dtype=np.float32,
        )
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        sample_positions = np.array([[0.2, 20.0, 0.2]], dtype=np.float32)
        sample_normals = np.zeros((1, 3), dtype=np.float32)

        actual, stats = trace_pmr_visibility_sh_tinybvh(
            positions,
            indices,
            sample_positions,
            sample_normals,
            ray_count=512,
            max_distance=0.5,
            self_bias=0.001,
            layout="auto",
        )
        expected = pmr.project_pmr_visibility_sh(
            np.ones((1, 512), dtype=np.float32),
            pmr.make_pmr_ray_directions(512),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        self.assertEqual(stats.visible_ray_count, 512)

    def test_vertex_visibility_cone_lines_encode_axis_aperture_and_rim(self):
        mesh = _mesh(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0, 1, 2]],
        )
        model = _model([mesh])
        cones = np.tile(
            np.array([0.0, 0.0, 1.0, np.pi * 0.5, 1.0], dtype=np.float32),
            (3, 1),
        )

        lines = build_visibility_cone_line_segments(
            model,
            [cones],
            cone_length=1.0,
            rim_segments=4,
        )

        self.assertEqual(lines.cone_count, 3)
        self.assertEqual(lines.invalid_direction_count, 0)
        self.assertEqual(lines.starts.shape, (27, 3))
        np.testing.assert_allclose(lines.starts[0], mesh.positions[0], atol=0.0)
        np.testing.assert_allclose(lines.ends[0], np.array([0.0, 0.0, 1.0]), atol=1e-7)
        np.testing.assert_allclose(lines.ends[1], np.array([1.0, 0.0, 0.0]), atol=1e-6)
        self.assertEqual(int(lines.widths[0]), 2)

    def test_interactive_visibility_cone_viewer_embeds_line_geometry(self):
        mesh = _mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
        )
        model = _model([mesh])
        colors = [np.full((3, 1), 0.5, dtype=np.float32)]
        cones = [
            np.tile(
                np.array([0.0, 0.0, 1.0, np.pi * 0.25, 1.8], dtype=np.float32),
                (3, 1),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cones.html"
            export_visibility_cone_viewer(model, colors, cones, output, cone_length=0.2, rim_segments=4)
            document = output.read_text(encoding="utf-8")

        self.assertIn("Vertex Visibility Cones", document)
        self.assertIn('id="cone-all"', document)
        self.assertIn('id="cone-surface"', document)
        self.assertIn('id="cone-wire"', document)
        self.assertIn('id="cone-xray"', document)
        self.assertIn('id="cone-diagram"', document)
        self.assertNotIn('const coneLineData = {"positions":[],"colors":[]}', document)
        payload_match = re.search(r"const coneInstanceData = (.*);", document)
        self.assertIsNotNone(payload_match)
        payload = json.loads(payload_match.group(1))
        self.assertEqual(len(payload["positions"]), 9)
        self.assertAlmostEqual(float(payload["length"]), 0.2, places=7)
        self.assertAlmostEqual(float(payload["parameters"][0]), np.pi * 0.25, places=6)
        self.assertAlmostEqual(float(payload["parameters"][1]), 1.8, places=6)
        self.assertIn("gl.drawArraysInstanced", document)
        self.assertIn("updateSelectedConeLines", document)

    def test_slang_viewer_vertex_buffers_preserve_order_and_clamp_pmr_cone(self):
        mesh0 = _mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
        )
        mesh1 = _mesh(
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            [[0, 1, 2]],
        )
        model = _model([mesh0, mesh1])
        values = [
            np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
            np.array([[0.4], [0.5], [0.6]], dtype=np.float32),
        ]
        cones = [
            np.tile(np.array([0.0, 0.0, 1.0, 2.0, 1.8], dtype=np.float32), (3, 1)),
            np.tile(np.array([0.0, 1.0, 0.0, 1.1, 0.4], dtype=np.float32), (3, 1)),
        ]

        packed_values = _vertex_value_float4(model, values)
        cone0, cone1 = _vertex_cone_float4(model, cones)

        np.testing.assert_allclose(packed_values[:, 0], np.arange(0.1, 0.7, 0.1), atol=1e-6)
        np.testing.assert_allclose(packed_values[:, 0], packed_values[:, 1], atol=0.0)
        np.testing.assert_allclose(cone0[:3, 3], HALF_PI, atol=1e-6)
        np.testing.assert_allclose(cone0[3:, 3], 1.1, atol=1e-6)
        np.testing.assert_allclose(cone1[:3, 0], 1.0, atol=1e-6)

    def test_pmr_mass_matrix_matches_analytic_triangle_integral(self):
        mesh = _mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
        )
        proxy = pmr.build_pmr_mesh_proxy(_model([mesh]))

        matrix = pmr.build_pmr_sparse_matrix(proxy, edge_regularization=0.0).toarray()
        expected = np.full((3, 3), 0.5 / 12.0, dtype=np.float64)
        np.fill_diagonal(expected, 0.5 / 6.0)
        np.testing.assert_allclose(matrix, expected, atol=1e-12)

    def test_pmr_constant_sample_sh_reconstructs_at_vertices(self):
        mesh = _mesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
        )
        sampling = pmr.sample_model_surface_pmr(_model([mesh]), samples_per_triangle=4)
        constant = np.linspace(-0.5, 1.0, 16, dtype=np.float32)
        sample_sh = np.tile(constant, (sampling.samples.positions.shape[0], 1))

        vertex_sh = pmr.fit_pmr_vertex_sh(sampling, sample_sh, edge_regularization=0.0)

        np.testing.assert_allclose(vertex_sh, np.tile(constant, (3, 1)), atol=2e-7)

    def test_pmr_sh_cone_fit_recovers_axis_aperture_and_scale(self):
        aperture = np.pi * 0.5
        scale = 0.7
        zonal_sh = np.zeros((16,), dtype=np.float32)
        for coefficient_index in (0, 2, 6, 12):
            zonal_sh[coefficient_index] = pmr._pmr_cone_coefficient(aperture, coefficient_index) * scale

        cones = pmr.pmr_vertex_sh_to_cones(
            zonal_sh[None, :],
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        )

        np.testing.assert_allclose(cones[0, :3], np.array([0.0, 0.0, 1.0]), atol=2e-6)
        self.assertAlmostEqual(float(cones[0, 3]), aperture, places=5)
        self.assertAlmostEqual(float(cones[0, 4]), scale, places=5)

        blocked = pmr.pmr_vertex_sh_to_cones(
            np.zeros((1, 16), dtype=np.float32),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(blocked[0, :3], np.zeros((3,), dtype=np.float32))
        self.assertAlmostEqual(float(blocked[0, 3]), 0.001, places=7)
        self.assertAlmostEqual(float(blocked[0, 4]), 1.0, places=7)

    def test_pmr_native_solver_and_cone_fit_match_python_reference(self):
        mesh = _mesh(
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            indices=[[0, 1, 2], [2, 1, 3]],
        )
        sampling = pmr.sample_model_surface_pmr(_model([mesh]), samples_per_triangle=16)
        rng = np.random.default_rng(19)
        sample_sh = rng.normal(0.0, 0.25, size=(32, 16)).astype(np.float32)
        sample_sh[:, 0] += np.float32(1.5)

        expected_sh = pmr.fit_pmr_vertex_sh(sampling, sample_sh, edge_regularization=0.05)
        native_sh = bake_pmr_visibility_sh_least_squares(
            sampling.proxy.positions,
            sampling.proxy.triangles,
            sampling.proxy.triangle_areas,
            sampling.samples_per_triangle,
            sampling.samples.barycentrics,
            sample_sh,
            edge_regularization=0.05,
        )
        np.testing.assert_allclose(native_sh, expected_sh, rtol=2e-5, atol=2e-6)

        expected_cones = pmr.pmr_vertex_sh_to_cones(expected_sh, sampling.proxy.normals)
        native_cones = pmr_sh_to_cones_native(native_sh, sampling.proxy.normals)
        np.testing.assert_allclose(native_cones, expected_cones, rtol=2e-5, atol=2e-5)

    def test_pmr_texcoord_encoding_preserves_full_sphere_aperture(self):
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        tangents = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        aperture = np.array([np.pi - 0.001], dtype=np.float32)
        scale = np.array([1.2], dtype=np.float32)

        encoded = encode_visibility_cone_texcoord2(
            normals,
            aperture,
            scale,
            normals,
            tangents,
            clamp_cone=False,
        )

        self.assertAlmostEqual(float(encoded[0, 2]), float(aperture[0]), places=6)
        self.assertAlmostEqual(float(encoded[0, 3]), 1.2, places=6)

    def test_python_least_squares_matches_native_reference(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        sample_triangles = np.array([0, 0, 1, 1], dtype=np.uint32)
        bary = np.array(
            [
                [0.6, 0.2, 0.2],
                [0.1, 0.7, 0.2],
                [0.2, 0.4, 0.4],
                [0.1, 0.2, 0.7],
            ],
            dtype=np.float32,
        )
        values = np.array(
            [
                [0.2, 0.1, 0.9, 1.0],
                [0.4, 0.3, 0.7, 0.8],
                [0.7, 0.2, 0.3, 0.6],
                [0.9, 0.8, 0.1, 0.5],
            ],
            dtype=np.float32,
        )

        python_result = bake_least_squares_python(positions, indices, sample_triangles, bary, values)
        native_result = bake_least_squares(positions, indices, sample_triangles, bary, values)

        np.testing.assert_allclose(python_result, native_result, atol=3e-5, rtol=3e-5)

    def test_native_visibility_fit_matches_python_encoding(self):
        mesh = _mesh(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
            uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        )
        tangents = compute_mesh_tangents(mesh)
        sample_triangles = np.array([0, 0, 0], dtype=np.uint32)
        bary = np.eye(3, dtype=np.float32)
        raw_sample_cones = np.array(
            [
                [0.0, 0.0, 1.0, HALF_PI, 1.0],
                [0.3, 0.1, 0.95, 0.8, 0.5],
                [-0.2, 0.4, 0.89, 1.2, 0.7],
            ],
            dtype=np.float32,
        )

        python_raw = bake_least_squares_python(mesh.positions, mesh.indices, sample_triangles, bary, raw_sample_cones)
        python_raw[:, :3] /= np.linalg.norm(python_raw[:, :3], axis=1, keepdims=True)
        python_raw[:, 3] = np.clip(python_raw[:, 3], 0.0, HALF_PI)
        python_raw[:, 4] = np.clip(python_raw[:, 4], 0.0, 1.0)
        python_encoded = encode_visibility_cone_texcoord2(
            python_raw[:, :3],
            python_raw[:, 3],
            python_raw[:, 4],
            mesh.normals,
            tangents,
        )

        native_raw, native_encoded = bake_visibility_least_squares(
            mesh.positions,
            mesh.normals,
            tangents,
            mesh.indices,
            sample_triangles,
            bary,
            raw_sample_cones,
        )

        np.testing.assert_allclose(native_raw, python_raw, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(native_encoded, python_encoded, atol=3e-5, rtol=3e-5)

    def test_fit_visibility_cones_native_backend_matches_python_backend(self):
        mesh = _mesh(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
            uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        )
        model = _model([mesh])
        samples = _with_sample_positions(model, _single_sample_on_triangle(bary=np.eye(3, dtype=np.float32)))
        directions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.25, 0.15, 0.96],
                [-0.2, 0.35, 0.92],
            ],
            dtype=np.float32,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        cones = VisibilitySampleCones(
            directions=directions,
            aperture_radians=np.array([HALF_PI, 0.9, 1.1], dtype=np.float32),
            scale=np.array([1.0, 0.6, 0.75], dtype=np.float32),
            visible_fraction=np.array([1.0, 0.4, 0.55], dtype=np.float32),
        )

        python_result = fit_visibility_cones(model, samples, cones, fit_backend="python")
        native_result = fit_visibility_cones(model, samples, cones, fit_backend="native")

        np.testing.assert_allclose(native_result.vertex_cones[0], python_result.vertex_cones[0], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(native_result.encoded_texcoord2[0], python_result.encoded_texcoord2[0], atol=3e-5, rtol=3e-5)

    def test_unconstrained_visibility_vertices_fall_back_to_unoccluded(self):
        mesh = _mesh(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [[0, 1, 2], [1, 3, 2]],
        )
        tangents = compute_mesh_tangents(mesh)
        sample_triangles = np.array([0], dtype=np.uint32)
        bary = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
        raw_sample_cones = np.array([[0.0, 0.0, 1.0, HALF_PI, 1.0]], dtype=np.float32)

        native_raw, native_encoded = bake_visibility_least_squares(
            mesh.positions,
            mesh.normals,
            tangents,
            mesh.indices,
            sample_triangles,
            bary,
            raw_sample_cones,
        )

        self.assertAlmostEqual(float(native_raw[3, 3]), HALF_PI, places=6)
        self.assertAlmostEqual(float(native_raw[3, 4]), 1.0, places=6)
        self.assertAlmostEqual(float(native_encoded[3, 2] / HALF_PI * native_encoded[3, 3]), 1.0, places=6)

    def test_fit_visibility_unconstrained_vertices_are_not_black(self):
        mesh = _mesh(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [[0, 1, 2], [1, 3, 2]],
        )
        model = _model([mesh])
        samples = _with_sample_positions(model, _single_sample_on_triangle(bary=np.array([[0.2, 0.3, 0.5]], dtype=np.float32)))
        cones = VisibilitySampleCones(
            directions=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            aperture_radians=np.array([HALF_PI], dtype=np.float32),
            scale=np.array([1.0], dtype=np.float32),
            visible_fraction=np.array([1.0], dtype=np.float32),
        )

        for backend in ("python", "native"):
            result = fit_visibility_cones(model, samples, cones, fit_backend=backend)
            ao = np.clip(result.encoded_texcoord2[0][:, 2] / HALF_PI, 0.0, 1.0) * result.encoded_texcoord2[0][:, 3]
            self.assertAlmostEqual(float(ao[3]), 1.0, places=6)

    def test_unoccluded_triangle_visibility_is_full_hemisphere(self):
        mesh = _mesh(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
            uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        )
        model = _model([mesh])
        samples = _with_sample_positions(model, _single_sample_on_triangle())

        cones = sample_visibility_cones_python(model, samples, ray_count=96, self_bias=1e-4)

        self.assertAlmostEqual(float(cones.visible_fraction[0]), 1.0, places=6)
        self.assertAlmostEqual(float(cones.aperture_radians[0]), HALF_PI, places=6)
        self.assertAlmostEqual(float(cones.scale[0]), 1.0, places=6)
        self.assertGreater(float(np.dot(cones.directions[0], np.array([0.0, 0.0, 1.0], dtype=np.float32))), 0.999)

    def test_vertical_blocker_bends_visibility_away_from_blocked_half_space(self):
        positions = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, -100.0, 0.0],
                [0.05, 100.0, 0.0],
                [0.05, -100.0, 100.0],
                [0.05, 100.0, 100.0],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        mesh = _mesh(
            positions,
            [[0, 1, 2], [3, 4, 5], [5, 4, 6]],
            normals=normals,
            uvs=np.zeros((positions.shape[0], 2), dtype=np.float32),
        )
        model = _model([mesh])
        samples = _with_sample_positions(model, _single_sample_on_triangle())

        cones = sample_visibility_cones_python(model, samples, ray_count=256, self_bias=1e-4)

        self.assertLess(float(cones.visible_fraction[0]), 0.62)
        self.assertGreater(float(cones.visible_fraction[0]), 0.42)
        self.assertLess(float(cones.directions[0, 0]), -0.2)

    def test_visibility_bake_outputs_gdc_texcoord2_channels(self):
        mesh = _mesh(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0, 1, 2]],
            uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        )
        model = _model([mesh])
        bary = np.eye(3, dtype=np.float32)
        samples = _with_sample_positions(model, _single_sample_on_triangle(bary=bary))

        result = bake_visibility_cones_python(model, samples=samples, visibility_ray_count=48, self_bias=1e-4)

        self.assertEqual(len(result.encoded_texcoord2), 1)
        encoded = result.encoded_texcoord2[0]
        self.assertEqual(encoded.shape, (3, 4))
        np.testing.assert_allclose(encoded[:, 2], HALF_PI, atol=1e-6)
        np.testing.assert_allclose(encoded[:, 3], 1.0, atol=1e-6)
        decoded = decode_visibility_cone_texcoord2(encoded, mesh.normals, compute_mesh_tangents(mesh))
        self.assertTrue(np.all(decoded.directions[:, 2] > 0.999))


if __name__ == "__main__":
    unittest.main()
