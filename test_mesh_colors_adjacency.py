import unittest
from pathlib import Path

import numpy as np
import slangpy as spy

from material import Material
from mesh import Mesh
from mesh_colors import MeshColorsLayout
from mesh_colors_adjacency import (
    MESH_COLORS_INVALID_ADJACENCY,
    MeshColorsEdgeAdjacency,
    build_triangle_adjacency,
    edge_barycentrics,
    edge_lattice_ij,
    remap_edge_parameter,
    triangle_edge_vertices,
)
from scene_node import SceneNode
from transform import Transform


PROJECT_DIR = Path(__file__).parent


def _valid_edge_count(adjacency) -> int:
    return sum(
        int(edge.valid)
        for face in adjacency.faces
        for edge in face.edges
    )


def _assert_reciprocal(test: unittest.TestCase, indices, adjacency) -> None:
    triangles = np.asarray(indices)
    for face_index, face in enumerate(adjacency.faces):
        for edge_index, edge in enumerate(face.edges):
            if not edge.valid:
                continue
            reverse = adjacency.faces[edge.adjacent_face][edge.adjacent_edge]
            test.assertTrue(reverse.valid)
            test.assertEqual(reverse.adjacent_face, face_index)
            test.assertEqual(reverse.adjacent_edge, edge_index)
            test.assertEqual(reverse.flip, edge.flip)

            source_vertices = triangle_edge_vertices(
                triangles[face_index], edge_index
            )
            adjacent_vertices = triangle_edge_vertices(
                triangles[edge.adjacent_face], edge.adjacent_edge
            )
            test.assertEqual(set(source_vertices), set(adjacent_vertices))
            for t in (0.0, 0.25, 0.5, 1.0):
                adjacent_t = remap_edge_parameter(t, edge)
                source_weights = {
                    source_vertices[0]: 1.0 - t,
                    source_vertices[1]: t,
                }
                adjacent_weights = {
                    adjacent_vertices[0]: 1.0 - adjacent_t,
                    adjacent_vertices[1]: adjacent_t,
                }
                test.assertEqual(source_weights.keys(), adjacent_weights.keys())
                for vertex in source_weights:
                    test.assertAlmostEqual(
                        source_weights[vertex], adjacent_weights[vertex]
                    )


class MeshColorsAdjacencyTopologyTests(unittest.TestCase):
    def test_isolated_triangle_is_all_boundary(self):
        adjacency = build_triangle_adjacency(
            np.array([[0, 1, 2]], dtype=np.uint32)
        )
        self.assertEqual(_valid_edge_count(adjacency), 0)
        self.assertEqual(adjacency.diagnostics.boundary_edge_count, 3)
        self.assertEqual(adjacency.diagnostics.manifold_edge_count, 0)

    def test_two_triangle_quad_has_one_reciprocal_edge(self):
        indices = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)
        adjacency = build_triangle_adjacency(indices)
        shared = adjacency.faces[0][1]
        self.assertEqual(shared.adjacent_face, 1)
        self.assertEqual(shared.adjacent_edge, 0)
        self.assertTrue(shared.flip)
        self.assertEqual(adjacency.diagnostics.boundary_edge_count, 4)
        self.assertEqual(adjacency.diagnostics.manifold_edge_count, 1)
        self.assertEqual(_valid_edge_count(adjacency), 2)
        _assert_reciprocal(self, indices, adjacency)

    def test_short_triangle_strip_links_only_consecutive_faces(self):
        indices = np.array(
            [[0, 1, 2], [2, 1, 3], [2, 3, 4]],
            dtype=np.uint32,
        )
        adjacency = build_triangle_adjacency(indices)
        self.assertEqual(adjacency.faces[0][1].adjacent_face, 1)
        self.assertEqual(adjacency.faces[1][2].adjacent_face, 2)
        self.assertEqual(adjacency.diagnostics.manifold_edge_count, 2)
        self.assertEqual(adjacency.diagnostics.boundary_edge_count, 5)
        self.assertEqual(_valid_edge_count(adjacency), 4)
        _assert_reciprocal(self, indices, adjacency)

    def test_open_and_closed_triangle_fans(self):
        open_indices = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4]],
            dtype=np.uint32,
        )
        open_adjacency = build_triangle_adjacency(open_indices)
        self.assertEqual(open_adjacency.diagnostics.manifold_edge_count, 2)
        self.assertEqual(open_adjacency.diagnostics.boundary_edge_count, 5)
        _assert_reciprocal(self, open_indices, open_adjacency)

        closed_indices = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
            dtype=np.uint32,
        )
        closed_adjacency = build_triangle_adjacency(closed_indices)
        self.assertEqual(closed_adjacency.diagnostics.manifold_edge_count, 4)
        self.assertEqual(closed_adjacency.diagnostics.boundary_edge_count, 4)
        self.assertTrue(
            all(
                sum(int(edge.valid) for edge in face.edges) == 2
                for face in closed_adjacency.faces
            )
        )
        traversal = []
        face_index = 0
        for _ in range(4):
            traversal.append(face_index)
            face_index = closed_adjacency.faces[face_index][2].adjacent_face
        self.assertEqual(traversal, [0, 1, 2, 3])
        self.assertEqual(face_index, 0)
        _assert_reciprocal(self, closed_indices, closed_adjacency)

    def test_closed_tetrahedron_has_no_boundaries(self):
        indices = np.array(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.uint32,
        )
        adjacency = build_triangle_adjacency(indices)
        self.assertEqual(adjacency.diagnostics.boundary_edge_count, 0)
        self.assertEqual(adjacency.diagnostics.manifold_edge_count, 6)
        self.assertEqual(adjacency.diagnostics.orientation_anomaly_count, 0)
        self.assertEqual(_valid_edge_count(adjacency), 12)
        _assert_reciprocal(self, indices, adjacency)

    def test_inconsistent_winding_is_paired_without_parameter_flip(self):
        indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)
        adjacency = build_triangle_adjacency(indices)
        shared = adjacency.faces[0][1]
        self.assertTrue(shared.valid)
        self.assertFalse(shared.flip)
        self.assertEqual(adjacency.diagnostics.orientation_anomaly_count, 1)
        _assert_reciprocal(self, indices, adjacency)

    def test_split_index_seam_stays_disconnected(self):
        adjacency = build_triangle_adjacency(
            np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        )
        self.assertEqual(_valid_edge_count(adjacency), 0)
        self.assertEqual(adjacency.diagnostics.boundary_edge_count, 6)

    def test_non_manifold_edge_is_diagnosed_and_left_invalid(self):
        indices = np.array(
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            dtype=np.uint32,
        )
        adjacency = build_triangle_adjacency(indices)
        self.assertEqual(adjacency.diagnostics.non_manifold_edge_count, 1)
        self.assertEqual(adjacency.diagnostics.boundary_edge_count, 6)
        self.assertFalse(adjacency.faces[0][0].valid)
        self.assertFalse(adjacency.faces[1][0].valid)
        self.assertFalse(adjacency.faces[2][0].valid)

    def test_degenerate_face_is_diagnosed_and_disconnected(self):
        adjacency = build_triangle_adjacency(
            np.array([[0, 0, 1], [1, 2, 3]], dtype=np.uint32)
        )
        self.assertEqual(adjacency.diagnostics.degenerate_face_count, 1)
        self.assertEqual(adjacency.diagnostics.boundary_edge_count, 3)
        self.assertEqual(_valid_edge_count(adjacency), 0)

    def test_edge_coordinates_match_mesh_colors_lattice_corners(self):
        expected_barycentrics = {
            (0, 0.0): (1.0, 0.0, 0.0),
            (0, 1.0): (0.0, 1.0, 0.0),
            (1, 0.0): (0.0, 1.0, 0.0),
            (1, 1.0): (0.0, 0.0, 1.0),
            (2, 0.0): (0.0, 0.0, 1.0),
            (2, 1.0): (1.0, 0.0, 0.0),
        }
        for (edge, t), expected in expected_barycentrics.items():
            self.assertEqual(edge_barycentrics(edge, t), expected)
            barycentrics = edge_barycentrics(edge, t)
            self.assertEqual(
                edge_lattice_ij(edge, t, 8),
                (barycentrics[0] * 8, barycentrics[1] * 8),
            )

    def test_pack_round_trip(self):
        entries = (
            MeshColorsEdgeAdjacency(),
            MeshColorsEdgeAdjacency(123, 2, False),
            MeshColorsEdgeAdjacency(456, 1, True),
        )
        for entry in entries:
            self.assertEqual(
                MeshColorsEdgeAdjacency.unpack(entry.pack()), entry
            )


class MeshColorsAdjacencyLayoutTests(unittest.TestCase):
    @staticmethod
    def _layout_for_mesh(mesh: Mesh, instance_count: int = 1) -> MeshColorsLayout:
        scene_node = SceneNode()
        mesh_id = scene_node.add_mesh(mesh)
        material_id = scene_node.add_material(Material())
        transform_id = scene_node.add_transform(Transform())
        for _ in range(instance_count):
            scene_node.add_instance(mesh_id, material_id, transform_id)
        return MeshColorsLayout.build(
            scene_node,
            texels_per_unit=1.0,
            min_resolution=1,
            max_resolution=16,
            max_total_texels=10_000,
        )

    @staticmethod
    def _two_instance_layout() -> MeshColorsLayout:
        vertices = np.zeros((4, 8), dtype=np.float32)
        vertices[:, 0:3] = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        mesh = Mesh(
            vertices,
            np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32),
        )
        return MeshColorsAdjacencyLayoutTests._layout_for_mesh(
            mesh, instance_count=2
        )

    def test_instances_repeat_mesh_local_adjacency_without_cross_links(self):
        layout = self._two_instance_layout()
        self.assertEqual(len(layout.face_infos), 4)
        self.assertEqual(len(layout.adjacency_infos), 4)
        self.assertEqual(layout.adjacency_diagnostics.manifold_edge_count, 1)
        self.assertEqual(layout.adjacency_diagnostics.boundary_edge_count, 4)

        for instance in layout.instance_infos:
            first_face = layout.adjacency_infos[instance.face_offset]
            neighbor = first_face[1]
            self.assertEqual(neighbor.adjacent_face, 1)
            global_neighbor = instance.face_offset + neighbor.adjacent_face
            self.assertGreaterEqual(global_neighbor, instance.face_offset)
            self.assertLess(
                global_neighbor, instance.face_offset + instance.face_count
            )

    def test_adjacency_is_independent_of_neighbor_face_resolution(self):
        vertices = np.zeros((4, 8), dtype=np.float32)
        vertices[:, 0:3] = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [8.0, 8.0, 0.0],
            ],
            dtype=np.float32,
        )
        mesh = Mesh(
            vertices,
            np.array([[0, 1, 2], [1, 0, 3]], dtype=np.uint32),
        )
        layout = self._layout_for_mesh(mesh)
        self.assertEqual(layout.face_infos[0].resolution, 2)
        self.assertEqual(layout.face_infos[1].resolution, 16)
        adjacency = layout.adjacency_infos[0][0]
        self.assertEqual(adjacency.adjacent_face, 1)
        self.assertEqual(adjacency.adjacent_edge, 0)
        self.assertTrue(adjacency.flip)
        source_t = 0.25
        adjacent_t = remap_edge_parameter(source_t, adjacency)
        self.assertEqual(
            edge_lattice_ij(0, source_t, 2),
            (1.5, 0.5),
        )
        self.assertEqual(
            edge_lattice_ij(0, adjacent_t, 16),
            (4.0, 12.0),
        )


class MeshColorsAdjacencyGpuTests(unittest.TestCase):
    def test_python_pack_matches_slang_decode(self):
        adjacency = build_triangle_adjacency(
            np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)
        )
        device = spy.Device(
            enable_debug_layers=False,
            compiler_options={"include_paths": [PROJECT_DIR]},
        )
        program = device.load_program(
            "test_mesh_colors_adjacency.slang", ["compute_main"]
        )
        pipeline = device.create_compute_pipeline(program)
        packed = np.full(
            (9, 4),
            MESH_COLORS_INVALID_ADJACENCY,
            dtype=np.uint32,
        )
        packed[7:9] = adjacency.packed_uint4()
        adjacency_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            data=packed,
        )
        output = device.create_buffer(
            usage=spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource,
            struct_size=16,
            element_count=3,
        )
        command_encoder = device.create_command_encoder()
        with command_encoder.begin_compute_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.g_adjacency_infos = adjacency_buffer
            cursor.g_output = output
            cursor.g_face_offset = 7
            pass_encoder.dispatch(thread_count=[3, 1, 1])
        device.submit_command_buffer(command_encoder.finish())
        device.wait_for_idle()

        decoded = (
            np.asarray(output.to_numpy())
            .view(np.uint32)
            .reshape(-1, 4)
        )
        expected = np.array(
            [
                [MESH_COLORS_INVALID_ADJACENCY, 0, 0, 0],
                [8, 0, 1, 1],
                [MESH_COLORS_INVALID_ADJACENCY, 0, 0, 0],
            ],
            dtype=np.uint32,
        )
        np.testing.assert_array_equal(decoded, expected)


if __name__ == "__main__":
    unittest.main()
