import unittest

import numpy as np

from scene_node import SceneNode
from surface_probes import (
    SURFACE_PROBE_FLAG_PROTECTED,
    SURFACE_PROBE_INSTANCE_SHIFT,
    SURFACE_PROBE_INSTANCE_SIZE,
    SURFACE_PROBE_METADATA_SIZE,
    SURFACE_PROBE_NODE_SIZE,
    SURFACE_PROBE_RADIAL_DEPTH_DIM,
    SURFACE_PROBE_RADIAL_MOMENT_SIZE,
    SurfaceProbeLayout,
    _build_point_octree_python,
    adaptive_weighted_sample_elimination,
    surface_probe_hemi_oct_decode,
    surface_probe_hemi_oct_encode,
    weighted_sample_elimination,
)
from surface_probe_sampler import (
    adaptive_weighted_sample_elimination_cpp,
    build_point_octree_cpp,
    cpp_surface_probe_sampler_version,
    deficit_repair_cpp,
    deficit_repair_python,
    estimate_support_cpp,
    estimate_support_python,
    filter_audit_repair_candidates_cpp,
)
from surface_probe_path_tracing_renderer import (
    SURFACE_PROBE_DEBUG_VIEWS,
    SURFACE_PROBE_SELF_HIT_SIZE,
    SurfaceProbePathTracingRenderer,
)


class SurfaceProbeRendererConfigurationTests(unittest.TestCase):
    def test_only_active_debug_resources_are_allocated(self):
        self.assertEqual(SURFACE_PROBE_SELF_HIT_SIZE, 4)
        self.assertEqual(SURFACE_PROBE_RADIAL_DEPTH_DIM, 4)
        self.assertEqual(SURFACE_PROBE_RADIAL_MOMENT_SIZE, 64)
        self.assertEqual(
            SURFACE_PROBE_DEBUG_VIEWS,
            (
                "Beauty",
                "Gather Count",
                "Support f(x)",
                "Density m(x)",
                "Vertex Fallback Weight",
                "Probe Self-hit Rate",
                "Vertex Lighting",
                "Vertex Confidence",
            ),
        )


class SurfaceProbeRadialDepthTests(unittest.TestCase):
    def test_hemi_oct_canonical_directions_cover_the_square(self):
        directions = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            surface_probe_hemi_oct_encode(directions),
            np.asarray(
                [
                    [0.5, 0.5],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_hemi_oct_round_trip_preserves_hemisphere_directions(self):
        rng = np.random.default_rng(20260728)
        directions = rng.normal(size=(4096, 3)).astype(np.float32)
        directions[:, 2] = np.abs(directions[:, 2])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        decoded = surface_probe_hemi_oct_decode(
            surface_probe_hemi_oct_encode(directions)
        )
        np.testing.assert_allclose(decoded, directions, atol=2e-6)


class PointOctreeParityTests(unittest.TestCase):
    def test_cpp_matches_python_node_layout_and_probe_order(self):
        rng = np.random.default_rng(1234)
        positions = rng.uniform(-2.0, 3.0, size=(4097, 3)).astype(np.float32)
        # Exercise Python's strict `position > float32(center)` boundary rule.
        positions[:8] = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -2.0],
                [0.5, -2.0, 0.5],
                [0.5, -2.0, -2.0],
                [-2.0, 0.5, 0.5],
                [-2.0, 0.5, -2.0],
                [-2.0, -2.0, 0.5],
                [-2.0, -2.0, -2.0],
            ],
            dtype=np.float32,
        )
        python_result = _build_point_octree_python(
            positions,
            leaf_capacity=8,
            max_depth=12,
        )
        profiles = []
        cpp_result = build_point_octree_cpp(
            positions,
            leaf_capacity=8,
            max_depth=12,
            profile_sink=profiles,
        )

        np.testing.assert_array_equal(cpp_result[0], python_result[0])
        np.testing.assert_array_equal(cpp_result[1], python_result[1])
        np.testing.assert_array_equal(cpp_result[2], python_result[2])
        self.assertEqual(cpp_result[3], python_result[3])
        self.assertEqual(len(profiles), 1)
        self.assertGreater(profiles[0].total_ms, 0.0)
        self.assertEqual(profiles[0].node_count, cpp_result[0].shape[0])

    def test_cpp_matches_degenerate_single_octant_chains(self):
        positions = np.zeros((33, 3), dtype=np.float32)
        python_result = _build_point_octree_python(
            positions,
            leaf_capacity=1,
            max_depth=5,
        )
        cpp_result = build_point_octree_cpp(
            positions,
            leaf_capacity=1,
            max_depth=5,
        )
        np.testing.assert_array_equal(cpp_result[0], python_result[0])
        np.testing.assert_array_equal(cpp_result[1], python_result[1])
        np.testing.assert_array_equal(cpp_result[2], python_result[2])
        self.assertEqual(cpp_result[3], python_result[3])


class WeightedSampleEliminationTests(unittest.TestCase):
    @staticmethod
    def _planar_grid(resolution: int) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
        x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
        positions = np.stack(
            (
                x.reshape(-1),
                np.zeros((x.size,), dtype=np.float32),
                y.reshape(-1),
            ),
            axis=1,
        )
        normals = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            (positions.shape[0], 1),
        )
        return positions, normals

    def test_elimination_is_deterministic_and_exact(self):
        positions, normals = self._planar_grid(20)

        first, radius = weighted_sample_elimination(
            positions,
            normals,
            64,
            surface_area=1.0,
            backend="cpp",
        )
        second, _ = weighted_sample_elimination(
            positions,
            normals,
            64,
            surface_area=1.0,
            backend="cpp",
        )

        self.assertEqual(first.shape, (64,))
        np.testing.assert_array_equal(first, second)
        self.assertGreater(radius, 0.0)
        self.assertEqual(np.unique(first).shape[0], 64)

    def test_cpp_backend_matches_python_spacing_quality(self):
        positions, normals = self._planar_grid(40)
        python_indices, python_radius = weighted_sample_elimination(
            positions,
            normals,
            256,
            surface_area=1.0,
            backend="python",
        )
        cpp_indices, cpp_radius = weighted_sample_elimination(
            positions,
            normals,
            256,
            surface_area=1.0,
            backend="cpp",
        )

        def mean_nearest_distance(indices):
            selected = positions[indices]
            distances = np.linalg.norm(
                selected[:, None, :] - selected[None, :, :], axis=2
            )
            np.fill_diagonal(distances, np.inf)
            return float(np.mean(np.min(distances, axis=1)))

        self.assertAlmostEqual(python_radius, cpp_radius, places=7)
        python_spacing = mean_nearest_distance(python_indices)
        cpp_spacing = mean_nearest_distance(cpp_indices)
        self.assertGreaterEqual(cpp_spacing, python_spacing * 0.95)

    def test_cpp_backend_reports_pinned_cycodebase_version(self):
        self.assertIn(
            "62da186e0b2f2d3673d1f18386c66caf5798cd9b",
            cpp_surface_probe_sampler_version(),
        )

    def test_unknown_backend_is_rejected(self):
        positions, normals = self._planar_grid(4)
        with self.assertRaisesRegex(ValueError, "Unknown.*backend"):
            weighted_sample_elimination(
                positions,
                normals,
                4,
                surface_area=1.0,
                backend="invalid",  # type: ignore[arg-type]
            )

    def test_opposite_surfaces_do_not_eliminate_each_other(self):
        # Same-side spacing must be < d_max so each sheet competes internally;
        # opposite sheets stay incompatible via normal gating.
        positions = np.array(
            [
                [-0.005, 0.0, 0.0],
                [0.005, 0.0, 0.0],
                [-0.005, 0.001, 0.0],
                [0.005, 0.001, 0.0],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        selected, _ = weighted_sample_elimination(
            positions,
            normals,
            2,
            surface_area=0.0004,
            backend="cpp",
        )
        selected_normals = normals[selected, 1]
        self.assertIn(1.0, selected_normals)
        self.assertIn(-1.0, selected_normals)


class AdaptiveWeightedSampleEliminationTests(unittest.TestCase):
    @staticmethod
    def _density_grid(resolution: int = 40):
        coordinates = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
        x, z = np.meshgrid(coordinates, coordinates, indexing="xy")
        positions = np.stack(
            (
                x.reshape(-1),
                np.zeros((x.size,), dtype=np.float32),
                z.reshape(-1),
            ),
            axis=1,
        )
        normals = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            (positions.shape[0], 1),
        )
        # Equal-area halves with mean relative density one and a 4:1 ratio.
        densities = np.where(
            positions[:, 0] < 0.5, 1.6, 0.4
        ).astype(np.float32)
        return positions, normals, densities

    def test_native_is_deterministic_and_favors_high_density_region(self):
        positions, normals, densities = self._density_grid()
        first, radius = adaptive_weighted_sample_elimination_cpp(
            positions,
            normals,
            densities,
            256,
            surface_area=1.0,
        )
        second, _ = adaptive_weighted_sample_elimination_cpp(
            positions,
            normals,
            densities,
            256,
            surface_area=1.0,
        )
        np.testing.assert_array_equal(first, second)
        high_count = int(np.count_nonzero(positions[first, 0] < 0.5))
        self.assertGreater(high_count, 180)
        self.assertLess(high_count, 220)
        self.assertGreater(radius, 0.0)

    def test_cpp_and_python_match_adaptive_population(self):
        positions, normals, densities = self._density_grid(20)
        cpp, cpp_radius = adaptive_weighted_sample_elimination(
            positions,
            normals,
            densities,
            64,
            surface_area=1.0,
            backend="cpp",
        )
        python, python_radius = adaptive_weighted_sample_elimination(
            positions,
            normals,
            densities,
            64,
            surface_area=1.0,
            backend="python",
        )
        cpp_high = int(np.count_nonzero(positions[cpp, 0] < 0.5))
        python_high = int(np.count_nonzero(positions[python, 0] < 0.5))
        self.assertLessEqual(abs(cpp_high - python_high), 2)
        self.assertAlmostEqual(cpp_radius, python_radius, places=6)

    def test_parallel_cpp_matches_python_density_at_scale(self):
        positions, normals, densities = self._density_grid(100)
        first, _ = adaptive_weighted_sample_elimination_cpp(
            positions,
            normals,
            densities,
            2000,
            surface_area=1.0,
        )
        second, _ = adaptive_weighted_sample_elimination_cpp(
            positions,
            normals,
            densities,
            2000,
            surface_area=1.0,
        )
        python, _ = adaptive_weighted_sample_elimination(
            positions,
            normals,
            densities,
            2000,
            surface_area=1.0,
            backend="python",
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(np.unique(first).shape[0], 2000)
        cpp_high = int(np.count_nonzero(positions[first, 0] < 0.5))
        python_high = int(np.count_nonzero(positions[python, 0] < 0.5))
        self.assertLessEqual(abs(cpp_high - python_high), 25)

    def test_normal_gate_still_isolates_opposite_sheets(self):
        positions = np.array(
            [
                [-0.01, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [-0.01, 0.001, 0.0],
                [0.01, 0.001, 0.0],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [[0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0]],
            dtype=np.float32,
        )
        selected, _ = adaptive_weighted_sample_elimination_cpp(
            positions,
            normals,
            np.ones((4,), dtype=np.float32),
            2,
            surface_area=0.0004,
        )
        self.assertIn(1.0, normals[selected, 1])
        self.assertIn(-1.0, normals[selected, 1])


class SurfaceDeficitRepairTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        base_positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        )
        base_normals = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (2, 1)
        )
        base_instances = np.zeros((2,), dtype=np.uint32)
        candidate_positions = np.array(
            [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.9, 0.0, 0.0]],
            dtype=np.float32,
        )
        candidate_normals = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1)
        )
        candidate_instances = np.zeros((3,), dtype=np.uint32)
        audit_positions = base_positions.copy()
        audit_normals = base_normals.copy()
        audit_instances = base_instances.copy()
        radii = np.array([0.5], dtype=np.float32)
        return (
            base_positions,
            base_normals,
            base_instances,
            candidate_positions,
            candidate_normals,
            candidate_instances,
            audit_positions,
            audit_normals,
            audit_instances,
            radii,
        )

    def test_cpp_matches_python_and_is_deterministic(self):
        inputs = self._inputs()
        python = deficit_repair_python(
            *inputs,
            min_gather_count=3,
            max_repair_count=3,
            normal_cosine_threshold=0.5,
        )
        cpp = deficit_repair_cpp(
            *inputs,
            min_gather_count=3,
            max_repair_count=3,
            normal_cosine_threshold=0.5,
        )
        np.testing.assert_array_equal(
            cpp.selected_candidate_indices,
            python.selected_candidate_indices,
        )
        np.testing.assert_array_equal(cpp.counts_before, python.counts_before)
        np.testing.assert_array_equal(cpp.counts_after, python.counts_after)
        np.testing.assert_allclose(cpp.ess_after, python.ess_after, rtol=1e-6)
        self.assertIsNotNone(cpp.profile)
        assert cpp.profile is not None
        self.assertGreater(cpp.profile.coverage_pair_count, 0)
        self.assertEqual(cpp.profile.affected_audit_count, 2)
        self.assertGreater(cpp.profile.worker_count, 0)
        self.assertGreaterEqual(cpp.profile.total_ms, 0.0)

    def test_parallel_repair_is_deterministic_at_scale(self):
        audit_axis = np.linspace(0.0, 1.0, 40, dtype=np.float32)
        audit_x, audit_z = np.meshgrid(audit_axis, audit_axis, indexing="xy")
        audit_positions = np.stack(
            (
                audit_x.reshape(-1),
                np.zeros((audit_x.size,), dtype=np.float32),
                audit_z.reshape(-1),
            ),
            axis=1,
        )
        base_axis = np.linspace(0.0, 1.0, 20, dtype=np.float32)
        base_x, base_z = np.meshgrid(base_axis, base_axis, indexing="xy")
        base_positions = np.stack(
            (
                base_x.reshape(-1),
                np.zeros((base_x.size,), dtype=np.float32),
                base_z.reshape(-1),
            ),
            axis=1,
        )
        base_normals = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            (base_positions.shape[0], 1),
        )
        audit_normals = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            (audit_positions.shape[0], 1),
        )
        inputs = (
            base_positions,
            base_normals,
            np.zeros((base_positions.shape[0],), dtype=np.uint32),
            audit_positions.copy(),
            audit_normals.copy(),
            np.zeros((audit_positions.shape[0],), dtype=np.uint32),
            audit_positions,
            audit_normals,
            np.zeros((audit_positions.shape[0],), dtype=np.uint32),
            np.array([0.06], dtype=np.float32),
        )
        first = deficit_repair_cpp(
            *inputs, min_gather_count=4, max_repair_count=100
        )
        second = deficit_repair_cpp(
            *inputs, min_gather_count=4, max_repair_count=100
        )
        np.testing.assert_array_equal(
            first.selected_candidate_indices,
            second.selected_candidate_indices,
        )
        np.testing.assert_array_equal(first.counts_before, second.counts_before)
        np.testing.assert_array_equal(first.counts_after, second.counts_after)
        assert first.profile is not None
        self.assertGreater(first.profile.worker_count, 1)
        self.assertLessEqual(
            first.profile.affected_audit_count, audit_positions.shape[0]
        )

    def test_budget_cap_and_irreparable_points_are_reported(self):
        result = deficit_repair_cpp(
            *self._inputs(),
            min_gather_count=4,
            max_repair_count=1,
            normal_cosine_threshold=0.5,
        )
        self.assertEqual(result.selected_candidate_indices.shape[0], 1)
        self.assertTrue(np.any(result.counts_after < 4))

    def test_lazy_heap_skips_zero_score_without_stopping(self):
        base_positions = np.array(
            [[9.9, 0.0, 0.0], [10.0, 0.0, 0.0], [10.1, 0.0, 0.0]],
            dtype=np.float32,
        )
        base_normals = np.tile([0.0, 1.0, 0.0], (3, 1)).astype(np.float32)
        instances = np.zeros((3,), dtype=np.uint32)
        candidate_positions = np.array(
            [[0.00, 0.0, 0.0], [0.02, 0.0, 0.0],
             [0.04, 0.0, 0.0], [0.06, 0.0, 0.0],
             [0.08, 0.0, 0.0], [10.2, 0.0, 0.0]],
            dtype=np.float32,
        )
        candidate_normals = np.tile([0.0, 1.0, 0.0], (6, 1)).astype(np.float32)
        result = deficit_repair_cpp(
            base_positions,
            base_normals,
            instances,
            candidate_positions,
            candidate_normals,
            np.zeros((6,), dtype=np.uint32),
            np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
            np.tile([0.0, 1.0, 0.0], (2, 1)).astype(np.float32),
            np.zeros((2,), dtype=np.uint32),
            np.array([0.5], dtype=np.float32),
            min_gather_count=4,
            max_repair_count=6,
            normal_cosine_threshold=0.5,
        )
        np.testing.assert_array_equal(result.counts_after, [4, 4])
        self.assertEqual(result.selected_candidate_indices.shape[0], 5)

    def test_no_deficit_adds_no_repairs(self):
        positions = np.array(
            [[-0.15, 0.0, 0.0], [-0.05, 0.0, 0.0],
             [0.05, 0.0, 0.0], [0.15, 0.0, 0.0]],
            dtype=np.float32,
        )
        normals = np.tile([0.0, 1.0, 0.0], (4, 1)).astype(np.float32)
        instances = np.zeros((4,), dtype=np.uint32)
        result = deficit_repair_cpp(
            positions,
            normals,
            instances,
            np.array([[0.3, 0.0, 0.0]], dtype=np.float32),
            normals[:1],
            instances[:1],
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            normals[:1],
            instances[:1],
            np.array([0.5], dtype=np.float32),
            min_gather_count=4,
            max_repair_count=1,
            normal_cosine_threshold=0.5,
        )
        self.assertEqual(result.selected_candidate_indices.shape[0], 0)

    def test_normal_and_plane_gates_are_respected(self):
        base_positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        up = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        down = -up
        instance = np.zeros((1,), dtype=np.uint32)
        result = deficit_repair_cpp(
            base_positions,
            up,
            instance,
            np.array(
                [[0.05, 0.0, 0.0], [0.0, 0.49, 0.0]], dtype=np.float32
            ),
            np.concatenate((down, up)),
            np.zeros((2,), dtype=np.uint32),
            base_positions,
            down,
            instance,
            np.array([0.5], dtype=np.float32),
            min_gather_count=1,
            max_repair_count=2,
            normal_cosine_threshold=0.5,
        )
        self.assertEqual(result.selected_candidate_indices.tolist(), [0])
        self.assertEqual(int(result.counts_after[0]), 1)

    def test_soft_normal_gate_keeps_sixty_degrees_and_rejects_eighty_five(self):
        angles = np.deg2rad(np.array([60.0, 85.0], dtype=np.float32))
        probe_normals = np.stack(
            (
                np.sin(angles),
                np.cos(angles),
                np.zeros_like(angles),
            ),
            axis=1,
        ).astype(np.float32)
        base_positions = np.array(
            [[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]], dtype=np.float32
        )
        empty_positions = np.zeros((0, 3), dtype=np.float32)
        empty_instances = np.zeros((0,), dtype=np.uint32)
        inputs = (
            base_positions,
            probe_normals,
            np.zeros((2,), dtype=np.uint32),
            empty_positions,
            empty_positions,
            empty_instances,
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            np.zeros((1,), dtype=np.uint32),
            np.array([0.5], dtype=np.float32),
        )
        python = deficit_repair_python(
            *inputs,
            min_gather_count=1,
            max_repair_count=0,
            normal_cosine_threshold=float(np.cos(np.deg2rad(45.0))),
        )
        cpp = deficit_repair_cpp(
            *inputs,
            min_gather_count=1,
            max_repair_count=0,
            normal_cosine_threshold=float(np.cos(np.deg2rad(45.0))),
        )
        self.assertEqual(int(python.counts_before[0]), 1)
        np.testing.assert_array_equal(cpp.counts_before, python.counts_before)
        np.testing.assert_allclose(
            cpp.weight_sums_before,
            python.weight_sums_before,
            rtol=1e-5,
        )


class SurfaceCandidateFilterTests(unittest.TestCase):
    def test_cell_normal_and_exact_duplicate_semantics(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        normals = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        result = filter_audit_repair_candidates_cpp(
            positions,
            normals,
            positions,
            normals,
            np.array([0], dtype=np.uint32),
            audit_cell_size=1.0,
            normal_cosine_threshold=0.5,
        )
        np.testing.assert_array_equal(result.audit_indices, [0, 3, 4])
        np.testing.assert_array_equal(result.repair_indices, [1, 3, 4])
        self.assertEqual(result.profile.audit_output_count, 3)
        self.assertEqual(result.profile.repair_output_count, 3)

    def test_parallel_filter_is_deterministic(self):
        rng = np.random.default_rng(7)
        positions = rng.uniform(-1.0, 1.0, size=(12_000, 3)).astype(
            np.float32
        )
        normals = rng.normal(size=(12_000, 3)).astype(np.float32)
        normals /= np.maximum(
            np.linalg.norm(normals, axis=1, keepdims=True), 1e-20
        )
        positions[6000:] = positions[:6000]
        normals[6000:] = normals[:6000]
        selected = np.arange(0, 6000, 3, dtype=np.uint32)
        first = filter_audit_repair_candidates_cpp(
            positions,
            normals,
            positions,
            normals,
            selected,
            audit_cell_size=0.05,
            normal_cosine_threshold=0.5,
        )
        second = filter_audit_repair_candidates_cpp(
            positions,
            normals,
            positions,
            normals,
            selected,
            audit_cell_size=0.05,
            normal_cosine_threshold=0.5,
        )
        np.testing.assert_array_equal(first.audit_indices, second.audit_indices)
        np.testing.assert_array_equal(
            first.repair_indices, second.repair_indices
        )
        self.assertGreater(first.profile.worker_count, 1)


class SurfaceSupportEstimateTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        coordinates = np.linspace(-2.0, 2.0, 41, dtype=np.float32)
        x, z = np.meshgrid(coordinates, coordinates, indexing="xy")
        positions = np.stack(
            (x.reshape(-1), np.zeros((x.size,), dtype=np.float32), z.reshape(-1)),
            axis=1,
        )
        normals = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            (positions.shape[0], 1),
        )
        instances = np.zeros((positions.shape[0],), dtype=np.uint32)
        area_weights = np.full(
            (positions.shape[0],), 0.01, dtype=np.float32
        )
        query_positions = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        query_normals = np.array(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        query_instances = np.zeros((3,), dtype=np.uint32)
        return (
            positions,
            normals,
            instances,
            area_weights,
            query_positions,
            query_normals,
            query_instances,
            np.array([1.0], dtype=np.float32),
        )

    def test_cpp_matches_python_and_detects_boundary_support(self):
        inputs = self._inputs()
        python = estimate_support_python(*inputs, max_density_multiplier=8.0)
        cpp = estimate_support_cpp(*inputs, max_density_multiplier=8.0)
        np.testing.assert_allclose(cpp.support_f, python.support_f, atol=1e-6)
        np.testing.assert_allclose(cpp.density_m, python.density_m, atol=1e-5)
        self.assertGreater(float(cpp.support_f[0]), 0.9)
        self.assertLess(float(cpp.support_f[1]), float(cpp.support_f[0]))
        self.assertEqual(float(cpp.support_f[2]), 0.0)
        self.assertEqual(float(cpp.density_m[2]), 8.0)

    def test_density_is_clamped_inverse_support(self):
        result = estimate_support_python(
            *self._inputs(), max_density_multiplier=4.0
        )
        expected = np.minimum(
            4.0, 1.0 / np.maximum(result.support_f, 0.25)
        )
        np.testing.assert_allclose(result.density_m, expected, atol=1e-6)


class SurfaceProbeLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene_node = SceneNode.demo()
        cls.layout = SurfaceProbeLayout.build(
            cls.scene_node,
            target_probe_count=80,
            oversample_factor=3,
            seed=7,
            leaf_capacity=8,
        )

    def test_budget_and_gpu_strides(self):
        self.assertEqual(self.layout.total_base_surface_site_count, 80)
        self.assertGreaterEqual(self.layout.total_candidate_count, 240)
        self.assertTrue(self.layout.adaptive_wse)
        self.assertLessEqual(self.layout.total_repair_surface_site_count, 24)
        self.assertEqual(
            self.layout.total_surface_site_count,
            80
            + self.layout.total_repair_surface_site_count
            + self.layout.total_protected_surface_site_count,
        )
        self.assertGreater(self.layout.total_audit_point_count, 0)
        self.assertGreater(self.layout.total_probe_count, 80)
        self.assertEqual(self.layout.total_vertex_anchor_site_count, 0)
        self.assertEqual(self.layout.total_vertex_anchor_probe_count, 0)
        self.assertEqual(
            len(self.layout.instance_infos),
            len(self.scene_node.instances),
        )
        self.assertEqual(
            self.layout.probes.dtype.itemsize,
            SURFACE_PROBE_METADATA_SIZE,
        )
        self.assertEqual(
            self.layout.nodes.dtype.itemsize * self.layout.nodes.shape[1],
            SURFACE_PROBE_NODE_SIZE,
        )
        self.assertEqual(
            self.layout.instance_gpu_data.dtype.itemsize,
            SURFACE_PROBE_INSTANCE_SIZE,
        )
        support_f = self.layout.probes["normal_side"][:, 3]
        self.assertTrue(np.all(np.isfinite(support_f)))
        self.assertTrue(np.all((support_f >= 0.0) & (support_f <= 1.0)))
        self.assertTrue(np.isfinite(self.layout.support_f_p10))
        self.assertTrue(np.isfinite(self.layout.support_f_p50))
        self.assertGreaterEqual(self.layout.density_m_p95, 1.0)
        self.assertLessEqual(
            self.layout.density_m_p95,
            self.layout.max_density_multiplier,
        )
        for instance_index, info in enumerate(self.layout.instance_infos):
            packed_instances = (
                self.layout.probes["meta"][
                    info.probe_offset : info.probe_offset + info.probe_count, 3
                ]
                >> SURFACE_PROBE_INSTANCE_SHIFT
            )
            self.assertTrue(np.all(packed_instances == instance_index))

    def test_compact_octree_ranges_are_valid(self):
        for info in self.layout.instance_infos:
            self.assertGreater(info.audit_point_count, 0)
            self.assertEqual(
                info.surface_site_count,
                info.base_surface_site_count
                + info.repair_surface_site_count
                + info.protected_surface_site_count,
            )
            node_end = info.node_offset + info.node_count
            probe_end = info.probe_offset + info.probe_count
            reconstruction_end = (
                info.probe_offset + info.reconstruction_probe_count
            )
            self.assertLessEqual(node_end, self.layout.nodes.shape[0])
            self.assertLessEqual(probe_end, self.layout.probes.shape[0])
            for node in self.layout.nodes[info.node_offset:node_end]:
                child_base, child_mask, probe_start, probe_count = (
                    int(value) for value in node
                )
                if child_mask:
                    self.assertGreaterEqual(child_base, info.node_offset)
                    self.assertLess(
                        child_base + child_mask.bit_count(),
                        node_end + 1,
                    )
                else:
                    self.assertGreaterEqual(probe_start, info.probe_offset)
                    self.assertLessEqual(
                        probe_start + probe_count,
                        reconstruction_end,
                    )

    def test_cpu_octree_query_finds_instance_probes(self):
        for instance_index, info in enumerate(self.layout.instance_infos):
            indices = self.layout.query_probe_indices(
                instance_index,
                info.root_center,
                radius=info.root_extent * 4.0,
            )
            expected = np.arange(
                info.probe_offset,
                info.probe_offset + info.reconstruction_probe_count,
                dtype=np.int64,
            )
            np.testing.assert_array_equal(np.sort(indices), expected)

            first_probe = info.probe_offset
            position = self.layout.probes["position_radius"][
                first_probe, :3
            ]
            tight = self.layout.query_probe_indices(
                instance_index,
                position,
                radius=1e-7,
            )
            self.assertIn(first_probe, tight)

    def test_triangle_vertex_anchor_map_references_anchor_probes(self):
        anchor_layout = SurfaceProbeLayout.build(
            self.scene_node,
            target_probe_count=80,
            oversample_factor=3,
            seed=7,
            build_vertex_anchors=True,
        )
        self.assertGreater(anchor_layout.total_vertex_anchor_site_count, 0)
        for instance_index, info in enumerate(anchor_layout.instance_infos):
            map_offset = anchor_layout.instance_gpu_data["params"][
                instance_index, 3
            ].view(np.uint32)
            mesh_id, _, _ = self.scene_node.instances[instance_index]
            triangle_count = self.scene_node.meshes[mesh_id].triangle_count
            records = anchor_layout.triangle_vertex_probes[
                int(map_offset) : int(map_offset) + triangle_count * 2
            ]
            valid = (records[:, 3] & np.uint32(1)) != 0
            self.assertTrue(np.any(valid))
            anchor_start = (
                info.probe_offset + info.reconstruction_probe_count
            )
            anchor_end = info.probe_offset + info.probe_count
            self.assertTrue(np.all(records[valid, :3] >= anchor_start))
            self.assertTrue(np.all(records[valid, :3] < anchor_end))

    def test_protected_closure_eliminates_unbudgeted_gather_deficits(self):
        options = dict(
            target_probe_count=8,
            oversample_factor=1,
            seed=7,
            kernel_radius_scale=0.5,
            repair_budget_ratio=0.0,
        )
        layout = SurfaceProbeLayout.build(self.scene_node, **options)
        self.assertEqual(layout.total_repair_surface_site_count, 0)
        self.assertGreater(layout.zero_gather_after_repair, 0)
        self.assertGreater(layout.deficit_point_count_after_repair, 0)
        self.assertGreater(layout.total_protected_surface_site_count, 0)
        self.assertEqual(layout.zero_gather_after, 0)
        self.assertEqual(layout.deficit_point_count_after, 0)
        self.assertEqual(layout.irreparable_audit_point_count, 0)
        self.assertEqual(layout.protected_stop_reason, "target_met")
        self.assertLessEqual(
            layout.total_protected_surface_site_count,
            layout.deficit_point_count_after_repair * 4,
        )
        protected = (
            layout.probes["meta"][:, 3] & SURFACE_PROBE_FLAG_PROTECTED
        ) != 0
        self.assertTrue(np.any(protected))
        self.assertEqual(layout.total_vertex_anchor_probe_count, 0)
        repeated = SurfaceProbeLayout.build(self.scene_node, **options)
        self.assertEqual(
            repeated.total_protected_surface_site_count,
            layout.total_protected_surface_site_count,
        )
        np.testing.assert_array_equal(repeated.probes, layout.probes)

    def test_deficit_audit_improves_or_preserves_coverage(self):
        for info in self.layout.instance_infos:
            self.assertLessEqual(info.zero_gather_after, info.zero_gather_before)
            self.assertLessEqual(
                info.deficit_point_count_after,
                info.deficit_point_count_after_repair,
            )
            self.assertLessEqual(
                info.deficit_point_count_after_repair,
                info.deficit_point_count_before,
            )
            self.assertLessEqual(
                info.deficit_point_count_after,
                info.deficit_point_count_before,
            )
            self.assertTrue(np.isfinite(info.ess_p50_before))
            self.assertTrue(np.isfinite(info.ess_p50_after))

    def test_zero_repair_ratio_preserves_exact_base_budget(self):
        layout = SurfaceProbeLayout.build(
            self.scene_node,
            target_probe_count=80,
            oversample_factor=3,
            seed=7,
            repair_budget_ratio=0.0,
        )
        self.assertEqual(layout.total_base_surface_site_count, 80)
        self.assertEqual(layout.total_repair_surface_site_count, 0)
        self.assertEqual(
            layout.total_surface_site_count,
            80 + layout.total_protected_surface_site_count,
        )
        self.assertEqual(layout.repair_surface_site_budget, 0)

    def test_adaptive_wse_reallocates_without_changing_base_budget(self):
        uniform = SurfaceProbeLayout.build(
            self.scene_node,
            target_probe_count=80,
            oversample_factor=3,
            seed=7,
            repair_budget_ratio=0.0,
            adaptive_wse=False,
        )
        self.assertFalse(uniform.adaptive_wse)
        self.assertEqual(uniform.total_base_surface_site_count, 80)
        self.assertEqual(self.layout.total_base_surface_site_count, 80)
        adaptive_counts = [
            info.base_surface_site_count for info in self.layout.instance_infos
        ]
        uniform_counts = [
            info.base_surface_site_count for info in uniform.instance_infos
        ]
        self.assertNotEqual(adaptive_counts, uniform_counts)


if __name__ == "__main__":
    unittest.main()
