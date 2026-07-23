import unittest
from types import SimpleNamespace

import numpy as np

from material import Material
from mesh import Mesh
from scene_node import SceneNode
from surface_probe_vertex_lighting import VertexLightingLayout
from surface_probes import (
    SURFACE_PROBE_DTYPE,
    SURFACE_PROBE_INSTANCE_DTYPE,
)
from transform import Transform


def _scene_with_mesh(mesh: Mesh, *, double_sided: bool = False) -> SceneNode:
    scene = SceneNode()
    mesh_id = scene.add_mesh(mesh)
    material_id = scene.add_material(Material(double_sided=double_sided))
    transform = Transform()
    transform.update_matrix()
    transform_id = scene.add_transform(transform)
    scene.add_instance(mesh_id, material_id, transform_id)
    return scene


def _probe_layout(
    triangle_indices: list[int],
    barycentrics: list[tuple[float, float, float]],
) -> SimpleNamespace:
    probes = np.zeros((len(triangle_indices),), dtype=SURFACE_PROBE_DTYPE)
    probes["normal_side"][:, 3] = 1.0
    probes["meta"][:, 0] = np.asarray(triangle_indices, dtype=np.uint32)
    probes["meta"][:, 1] = np.asarray(
        [barycentric[1] for barycentric in barycentrics], dtype=np.float32
    ).view(np.uint32)
    probes["meta"][:, 2] = np.asarray(
        [barycentric[2] for barycentric in barycentrics], dtype=np.float32
    ).view(np.uint32)
    instances = np.zeros((1,), dtype=SURFACE_PROBE_INSTANCE_DTYPE)
    instances["offsets"][0] = (0, 0, 0, len(triangle_indices))
    return SimpleNamespace(probes=probes, instance_gpu_data=instances)


def _project_reference(
    layout: VertexLightingLayout,
    probe_values: np.ndarray,
) -> np.ndarray:
    output = np.zeros((layout.vertex_count,), dtype=np.float64)
    for vertex in range(layout.vertex_count):
        begin = int(layout.projection_offsets[vertex])
        end = int(layout.projection_offsets[vertex + 1])
        samples = layout.projection_samples[begin:end]
        if samples.size == 0:
            continue
        weights = samples[:, 1].view(np.float32).astype(np.float64)
        output[vertex] = np.sum(
            weights * probe_values[samples[:, 0].astype(np.int64)]
        ) / np.sum(weights)
    return output


class VertexLightingLayoutTests(unittest.TestCase):
    def test_quad_builds_shared_topology_and_front_triangle_map(self):
        layout = VertexLightingLayout.build(
            _scene_with_mesh(Mesh.create_quad())
        )

        self.assertEqual(layout.vertex_count, 4)
        self.assertEqual(layout.edge_count, 10)
        self.assertEqual(layout.triangle_map.shape, (4, 4))
        np.testing.assert_array_equal(layout.triangle_map[0, 3], 1)
        np.testing.assert_array_equal(layout.triangle_map[1], 0)
        np.testing.assert_array_equal(layout.triangle_map[2, 3], 1)
        np.testing.assert_array_equal(layout.triangle_map[3], 0)
        self.assertEqual(int(layout.neighbor_offsets[-1]), 10)

    def test_double_sided_mesh_has_independent_front_and_back_graphs(self):
        layout = VertexLightingLayout.build(
            _scene_with_mesh(Mesh.create_quad(), double_sided=True)
        )

        self.assertEqual(layout.vertex_count, 8)
        self.assertEqual(layout.edge_count, 20)
        self.assertTrue(np.all(layout.triangle_map[:, 3] == 1))
        front_vertices = set(layout.triangle_map[0, :3])
        back_vertices = set(layout.triangle_map[1, :3])
        self.assertTrue(front_vertices.isdisjoint(back_vertices))
        for vertex in range(4):
            begin = int(layout.neighbor_offsets[vertex])
            end = int(layout.neighbor_offsets[vertex + 1])
            self.assertTrue(np.all(layout.neighbors[begin:end, 0] < 4))
        for vertex in range(4, 8):
            begin = int(layout.neighbor_offsets[vertex])
            end = int(layout.neighbor_offsets[vertex + 1])
            self.assertTrue(np.all(layout.neighbors[begin:end, 0] >= 4))

    def test_sliver_triangles_receive_stronger_regularization(self):
        equilateral = Mesh.create_triangle(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.8660254], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        sliver = Mesh.create_triangle(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.001, 0.0, 0.00001], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        good = VertexLightingLayout.build(_scene_with_mesh(equilateral))
        bad = VertexLightingLayout.build(_scene_with_mesh(sliver))

        self.assertLess(good.condition_mean, 1.01)
        self.assertGreater(bad.condition_mean, 10.0)

    def test_compatible_split_vertices_are_reconnected_across_uv_seam(self):
        vertices = np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [0, 1, 0, 0, 0, 1, 1, 0],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        layout = VertexLightingLayout.build(
            _scene_with_mesh(Mesh(vertices, indices))
        )

        self.assertEqual(layout.weld_edge_count, 2)
        self.assertEqual(layout.edge_count, 16)
        seam_neighbors = {
            tuple(map(int, edge))
            for edge in layout.neighbors[:, :1]
        }
        begin = int(layout.neighbor_offsets[1])
        end = int(layout.neighbor_offsets[2])
        self.assertIn(3, set(map(int, layout.neighbors[begin:end, 0])))
        self.assertTrue(seam_neighbors)

    def test_area_projection_preserves_a_constant_on_a_coarse_mesh(self):
        probes = _probe_layout(
            [0, 1],
            [(0.98, 0.01, 0.01), (0.02, 0.93, 0.05)],
        )
        layout = VertexLightingLayout.build(
            _scene_with_mesh(Mesh.create_quad()),
            probe_layout=probes,
        )

        self.assertEqual(layout.projection_sample_count, 6)
        np.testing.assert_allclose(
            _project_reference(layout, np.array([2.5, 2.5])),
            2.5,
            rtol=1e-6,
        )
        self.assertEqual(layout.zero_projection_vertex_count, 0)
        self.assertEqual(layout.partial_projection_vertex_count, 0)

    def test_missing_triangle_is_reported_as_zero_and_partial_projection(self):
        layout = VertexLightingLayout.build(
            _scene_with_mesh(Mesh.create_quad()),
            probe_layout=_probe_layout(
                [0],
                [(1.0 / 3.0,) * 3],
            ),
        )

        self.assertEqual(layout.zero_projection_vertex_count, 1)
        self.assertEqual(layout.partial_projection_vertex_count, 2)
        np.testing.assert_allclose(
            layout.vertex_position_area[:, 3],
            np.array([1.0, 2.0, 2.0, 1.0]) / 6.0,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(layout.vertex_normal[:, :3], axis=1),
            1.0,
            rtol=1e-6,
        )

    def test_repair_density_does_not_bias_area_projection(self):
        triangle_indices = [0] + [1] * 10
        barycentrics = [(1.0 / 3.0,) * 3] * len(triangle_indices)
        layout = VertexLightingLayout.build(
            _scene_with_mesh(Mesh.create_quad()),
            probe_layout=_probe_layout(triangle_indices, barycentrics),
        )
        projected = _project_reference(
            layout,
            np.array([1.0] + [3.0] * 10, dtype=np.float64),
        )

        triangle_vertices = layout.triangle_map[0::2, :3]
        shared = set(map(int, triangle_vertices[0])).intersection(
            map(int, triangle_vertices[1])
        )
        self.assertEqual(len(shared), 2)
        np.testing.assert_allclose(
            projected[list(shared)],
            2.0,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
