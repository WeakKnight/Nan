import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import slangpy as spy

from material import Material
from mesh import Mesh
from mesh_colors import MESH_COLORS_PAYLOAD_SIZE, MeshColorsLayout, colors_per_patch
from mesh_colors_adjacency import MESH_COLORS_ADJACENCY_FLIP_BIT
from mesh_colors_surface_filter import MeshColorsSurfaceFilter
from scene_node import SceneNode
from texture_space_path_tracing_renderer import TextureSpacePathTracingRenderer
from transform import Transform


PROJECT_DIR = Path(__file__).parent


def _scene_node(
    positions,
    indices,
    *,
    double_sided: bool = False,
) -> SceneNode:
    positions = np.asarray(positions, dtype=np.float32)
    vertices = np.zeros((len(positions), 8), dtype=np.float32)
    vertices[:, :3] = positions
    mesh = Mesh(vertices, np.asarray(indices, dtype=np.uint32))
    node = SceneNode()
    mesh_id = node.add_mesh(mesh)
    material_id = node.add_material(Material(double_sided=double_sided))
    transform_id = node.add_transform(Transform())
    node.add_instance(mesh_id, material_id, transform_id)
    return node


def _layout(node: SceneNode, resolution: int) -> MeshColorsLayout:
    return MeshColorsLayout.build(
        node,
        texels_per_unit=1.0,
        min_resolution=resolution,
        max_resolution=resolution,
        max_total_texels=100_000,
    )


def _payload(
    layout: MeshColorsLayout,
    values: np.ndarray | float,
    *,
    sample_count: int = 17,
) -> np.ndarray:
    data = np.zeros((layout.total_payload_count, 4), dtype=np.float32)
    data[:, :3] = values
    data.view(np.uint32)[:, 3] = np.uint32(sample_count)
    return data


class MeshColorsSurfaceFilterGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = spy.Device(
            enable_debug_layers=False,
            compiler_options={"include_paths": [PROJECT_DIR]},
        )

    def _run_filter(
        self,
        node: SceneNode,
        layout: MeshColorsLayout,
        payload: np.ndarray,
        *,
        pass_count: int = 1,
        spatial_sigma: float = 1.0,
        normal_sigma_degrees: float = 30.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        surface_filter = MeshColorsSurfaceFilter(
            self.device,
            SimpleNamespace(scene_node=node),
            layout,
        )
        source = self.device.create_buffer(
            usage=spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource,
            data=payload,
        )
        scratch = self.device.create_buffer(
            usage=spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource,
            struct_size=MESH_COLORS_PAYLOAD_SIZE,
            element_count=layout.total_payload_count,
        )
        command_encoder = self.device.create_command_encoder()
        destination = scratch
        for _ in range(pass_count):
            surface_filter.execute(
                command_encoder,
                source,
                destination,
                spatial_sigma=spatial_sigma,
                normal_sigma_radians=math.radians(normal_sigma_degrees),
            )
            source, destination = destination, source
        self.device.submit_command_buffer(command_encoder.finish())
        self.device.wait_for_idle()
        result = np.asarray(source.to_numpy()).view(np.float32).reshape(-1, 4)
        counts = np.asarray(source.to_numpy()).view(np.uint32).reshape(-1, 4)[:, 3]
        return result[:, :3].copy(), counts.copy()

    def _run_helper_shader(self, entry_point: str, cases: np.ndarray) -> np.ndarray:
        program = self.device.load_program(
            "test_mesh_colors_surface_filter.slang", [entry_point]
        )
        pipeline = self.device.create_compute_pipeline(program)
        case_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            data=cases,
        )
        output = self.device.create_buffer(
            usage=spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource,
            struct_size=16,
            element_count=max(1, len(cases)),
        )
        command_encoder = self.device.create_command_encoder()
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_cases = case_buffer
            cursor.g_output = output
            pass_encoder.dispatch(thread_count=[max(1, len(cases)), 1, 1])
        self.device.submit_command_buffer(command_encoder.finish())
        self.device.wait_for_idle()
        return np.asarray(output.to_numpy()).view(np.float32).reshape(-1, 4)

    def test_equal_and_unequal_resolution_remap_and_corner_skip(self):
        minus_one = np.asarray([-1], dtype=np.int32).view(np.uint32)[0]
        flip_edge_zero = np.uint32(MESH_COLORS_ADJACENCY_FLIP_BIT)
        cases = np.array(
            [
                [3, 2, 4, flip_edge_zero],
                [2, 2, 3, flip_edge_zero],
                [
                    minus_one,
                    0,
                    4,
                    flip_edge_zero | np.uint32(1 << 8),
                ],
            ],
            dtype=np.uint32,
        )
        decoded = self._run_helper_shader("compute_main", cases)
        np.testing.assert_allclose(decoded[0], [0.25, 0.5, 0.25, 1.0])
        np.testing.assert_allclose(
            decoded[0, :3] * 4.0, [1.0, 2.0, 1.0], atol=1e-6
        )
        np.testing.assert_allclose(
            decoded[1], [1 / 3, 1 / 3, 1 / 3, 1.0], atol=1e-6
        )
        self.assertFalse(
            np.allclose(
                decoded[1, :3] * 8.0,
                np.round(decoded[1, :3] * 8.0),
            )
        )
        self.assertEqual(decoded[2, 3], 0.0)

    def test_normal_weight_formula(self):
        cases = np.zeros((1, 4), dtype=np.uint32)
        decoded = self._run_helper_shader("normal_weight_main", cases)[0]
        self.assertAlmostEqual(float(decoded[0]), 1.0, places=6)
        self.assertAlmostEqual(float(decoded[1]), math.exp(-4.5), places=6)
        self.assertAlmostEqual(float(decoded[2]), math.exp(-1.125), places=6)

    def test_constant_field_preserved_on_boundary_strip_and_closed_fan(self):
        topologies = [
            (
                [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                [[0, 1, 2]],
            ),
            (
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0)],
                [[0, 1, 2], [2, 1, 3], [2, 3, 4]],
            ),
            (
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
                [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
            ),
        ]
        for positions, indices in topologies:
            with self.subTest(face_count=len(indices)):
                node = _scene_node(positions, indices)
                layout = _layout(node, 4)
                expected = np.array([2.0, 0.5, 4.0], dtype=np.float32)
                result, counts = self._run_filter(
                    node, layout, _payload(layout, expected), pass_count=4
                )
                np.testing.assert_allclose(
                    result,
                    np.broadcast_to(expected, result.shape),
                    atol=2e-6,
                )
                np.testing.assert_array_equal(counts, 17)

    def test_constant_field_preserved_across_unequal_resolution_edge(self):
        node = _scene_node(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (8, 8, 0)],
            [[0, 1, 2], [1, 0, 3]],
        )
        layout = MeshColorsLayout.build(
            node,
            texels_per_unit=1.0,
            min_resolution=1,
            max_resolution=16,
            max_total_texels=100_000,
        )
        self.assertNotEqual(
            layout.face_infos[0].resolution,
            layout.face_infos[1].resolution,
        )
        expected = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        result, _ = self._run_filter(
            node, layout, _payload(layout, expected), pass_count=3
        )
        np.testing.assert_allclose(
            result,
            np.broadcast_to(expected, result.shape),
            atol=2e-6,
        )

    def test_closed_fan_impulse_expands_by_at_most_one_face_ring_per_pass(self):
        node = _scene_node(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
        )
        layout = _layout(node, 2)
        values = np.zeros((layout.total_payload_count, 3), dtype=np.float32)
        first_face = layout.face_infos[0]
        values[first_face.address : first_face.address + colors_per_patch(2)] = 1.0

        one_pass, _ = self._run_filter(node, layout, _payload(layout, values))
        face_max = []
        for face in layout.face_infos:
            face_max.append(
                float(
                    np.max(
                        one_pass[
                            face.address : face.address + colors_per_patch(2)
                        ]
                    )
                )
            )
        self.assertGreater(face_max[0], 0.0)
        self.assertGreater(face_max[1], 0.0)
        self.assertEqual(face_max[2], 0.0)
        self.assertGreater(face_max[3], 0.0)

        two_pass, _ = self._run_filter(
            node, layout, _payload(layout, values), pass_count=2
        )
        face_two = layout.face_infos[2]
        self.assertGreater(
            float(
                np.max(
                    two_pass[
                        face_two.address : face_two.address
                        + colors_per_patch(2)
                    ]
                )
            ),
            0.0,
        )

    def test_split_seams_and_non_manifold_edges_do_not_connect(self):
        cases = [
            (
                [
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                ],
                [[0, 1, 2], [3, 4, 5]],
            ),
            (
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)],
                [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            ),
        ]
        for positions, indices in cases:
            with self.subTest(face_count=len(indices)):
                node = _scene_node(positions, indices)
                layout = _layout(node, 2)
                values = np.zeros(
                    (layout.total_payload_count, 3), dtype=np.float32
                )
                values[: colors_per_patch(2)] = 1.0
                result, _ = self._run_filter(
                    node, layout, _payload(layout, values), pass_count=3
                )
                self.assertEqual(
                    float(np.max(result[colors_per_patch(2) :])), 0.0
                )

    def test_sharp_fold_receives_substantially_less_cross_face_blur(self):
        indices = [[0, 1, 2], [1, 0, 3]]
        coplanar = _scene_node(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0)],
            indices,
        )
        folded = _scene_node(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            indices,
        )

        received = []
        for node in (coplanar, folded):
            layout = _layout(node, 4)
            values = np.zeros(
                (layout.total_payload_count, 3), dtype=np.float32
            )
            values[: colors_per_patch(4)] = 1.0
            result, _ = self._run_filter(
                node,
                layout,
                _payload(layout, values),
                normal_sigma_degrees=30.0,
            )
            received.append(float(np.max(result[colors_per_patch(4) :])))
        self.assertGreater(received[0], 0.1)
        self.assertLess(received[1], received[0] * 0.05)

    def test_front_and_back_payloads_and_counts_remain_independent(self):
        node = _scene_node(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [[0, 1, 2]],
            double_sided=True,
        )
        layout = _layout(node, 4)
        values = np.zeros((layout.total_payload_count, 3), dtype=np.float32)
        values[: layout.total_surface_texels] = 3.0
        payload = _payload(layout, values, sample_count=123)
        result, counts = self._run_filter(node, layout, payload, pass_count=5)
        np.testing.assert_allclose(
            result[: layout.total_surface_texels], 3.0, atol=2e-6
        )
        np.testing.assert_array_equal(
            result[layout.total_surface_texels :], 0.0
        )
        np.testing.assert_array_equal(counts, 123)

    def test_noise_variance_decreases_while_mean_remains_stable(self):
        node = _scene_node(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [[0, 1, 2]],
        )
        layout = _layout(node, 16)
        rng = np.random.default_rng(7)
        values = 2.0 + rng.normal(
            0.0, 0.5, size=(layout.total_payload_count, 3)
        ).astype(np.float32)
        result, _ = self._run_filter(
            node, layout, _payload(layout, values), pass_count=5
        )
        self.assertLess(float(np.var(result)), float(np.var(values)) * 0.2)
        self.assertLess(
            abs(float(np.mean(result)) - float(np.mean(values))),
            abs(float(np.mean(values))) * 0.03,
        )


class MeshColorsSurfaceFilterStateTests(unittest.TestCase):
    @staticmethod
    def _renderer() -> TextureSpacePathTracingRenderer:
        renderer = TextureSpacePathTracingRenderer.__new__(
            TextureSpacePathTracingRenderer
        )
        renderer.filter_before_save_check_box = None
        renderer.filter_pass_count_slider = None
        renderer.filter_spatial_sigma_slider = None
        renderer.filter_normal_angle_slider = None
        renderer._save_requested = True
        renderer._frozen_rgb9e5 = None
        renderer._release_frozen_buffer = False
        renderer._save_error = None
        renderer.reset_accumulator = False
        renderer.reset_texture_space = False
        renderer.texture_iteration = 3
        renderer.save_rgb9e5_button = None
        renderer.layout = SimpleNamespace(total_payload_count=123)
        return renderer

    def test_filter_enabled_ping_pongs_then_packs_latest_buffer(self):
        renderer = self._renderer()
        filter_calls = []
        pack_calls = []
        renderer.surface_filter = SimpleNamespace(
            execute=lambda encoder, source, destination, **kwargs: (
                filter_calls.append((source, destination))
            )
        )
        renderer.rgb9e5_packer = SimpleNamespace(
            execute=lambda encoder, source, destination, count: (
                pack_calls.append((source, destination, count))
            )
        )
        working = object()
        scratch = object()
        packed = object()
        self.assertTrue(
            renderer._execute_rgb9e5_save(
                object(), working, scratch, packed
            )
        )
        self.assertEqual(len(filter_calls), 5)
        self.assertEqual(
            filter_calls,
            [
                (working, scratch),
                (scratch, working),
                (working, scratch),
                (scratch, working),
                (working, scratch),
            ],
        )
        self.assertEqual(pack_calls, [(scratch, packed, 123)])
        self.assertIs(renderer._frozen_rgb9e5, packed)
        self.assertTrue(renderer.reset_accumulator)

    def test_filter_disabled_packs_working_buffer_and_reset_discards_frozen(self):
        renderer = self._renderer()
        renderer.filter_before_save_check_box = SimpleNamespace(value=False)
        packed_calls = []
        renderer.rgb9e5_packer = SimpleNamespace(
            execute=lambda encoder, source, destination, count: (
                packed_calls.append((source, destination, count))
            )
        )
        working = object()
        packed = object()
        self.assertTrue(
            renderer._execute_rgb9e5_save(
                object(), working, None, packed
            )
        )
        self.assertEqual(packed_calls, [(working, packed, 123)])

        renderer._discard_rgb9e5()
        self.assertIsNone(renderer._frozen_rgb9e5)
        self.assertTrue(renderer._release_frozen_buffer)


if __name__ == "__main__":
    unittest.main()
