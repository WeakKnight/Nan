from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import slangpy as spy

from render_data import RenderData
from scene import Scene
from scene_node import SceneNode
from surface_probe_path_tracer import SurfaceProbePathTracer
from surface_probes import (
    SURFACE_PROBE_FLAG_BACK_SIDE,
    SURFACE_PROBE_FLAG_VERTEX_ANCHOR,
    SurfaceProbeLayout,
)


VERTEX_LIGHTING_TARGET_KEY = "surface_probe_renderer.vertex_lighting_target"
VERTEX_LIGHTING_WORK_A_KEY = "surface_probe_renderer.vertex_lighting_work_a"
VERTEX_LIGHTING_WORK_B_KEY = "surface_probe_renderer.vertex_lighting_work_b"
VERTEX_LIGHTING_RGBM_KEY = "surface_probe_renderer.vertex_lighting_rgbm"


def _world_geometry(
    scene_node: SceneNode,
    instance_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_id, _, transform_id = scene_node.instances[instance_index]
    mesh = scene_node.meshes[mesh_id]
    transform = np.asarray(
        scene_node.transforms[transform_id].matrix.to_numpy(), dtype=np.float64
    )
    positions = (
        mesh.vertices[:, :3].astype(np.float64) @ transform[:3, :3].T
        + transform[:3, 3]
    )
    normal_matrix = np.linalg.inv(transform[:3, :3]).T
    normals = mesh.vertices[:, 3:6].astype(np.float64) @ normal_matrix.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    return positions, normals


@dataclass(frozen=True)
class VertexLightingLayout:
    """Per-instance render vertices plus a hard-edge-aware topology graph."""

    vertex_meta: np.ndarray
    vertex_position_area: np.ndarray
    vertex_normal: np.ndarray
    neighbor_offsets: np.ndarray
    neighbors: np.ndarray
    projection_offsets: np.ndarray
    projection_samples: np.ndarray
    triangle_map: np.ndarray
    instance_offsets: np.ndarray
    instance_vertex_counts: tuple[int, ...]
    weld_edge_count: int
    condition_mean: float
    condition_p95: float
    zero_projection_vertex_count: int
    partial_projection_vertex_count: int

    @property
    def vertex_count(self) -> int:
        return int(self.vertex_meta.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.neighbors.shape[0])

    @property
    def projection_sample_count(self) -> int:
        return int(self.projection_samples.shape[0])

    @classmethod
    def build(
        cls,
        scene_node: SceneNode,
        *,
        probe_layout: SurfaceProbeLayout | None = None,
        hard_edge_angle_degrees: float = 60.0,
        maximum_condition_multiplier: float = 32.0,
    ) -> "VertexLightingLayout":
        hard_edge_cosine = math.cos(math.radians(hard_edge_angle_degrees))
        maximum_condition_multiplier = max(
            1.0, float(maximum_condition_multiplier)
        )
        vertex_meta_arrays: list[np.ndarray] = []
        vertex_position_area_arrays: list[np.ndarray] = []
        vertex_normal_arrays: list[np.ndarray] = []
        neighbor_source_arrays: list[np.ndarray] = []
        neighbor_destination_arrays: list[np.ndarray] = []
        neighbor_weight_arrays: list[np.ndarray] = []
        projection_source_arrays: list[np.ndarray] = []
        projection_probe_arrays: list[np.ndarray] = []
        projection_weight_arrays: list[np.ndarray] = []
        projection_coverage_weight_arrays: list[np.ndarray] = []
        triangle_record_arrays: list[np.ndarray] = []
        instance_offsets: list[tuple[int, int]] = []
        instance_vertex_counts: list[int] = []
        condition_arrays: list[np.ndarray] = []
        vertex_cursor = 0
        triangle_cursor = 0
        weld_edge_count = 0

        for instance_index, (mesh_id, material_id, _) in enumerate(
            scene_node.instances
        ):
            mesh = scene_node.meshes[mesh_id]
            triangles = np.asarray(mesh.indices, dtype=np.int64)
            instance_offsets.append((triangle_cursor, mesh.triangle_count))
            triangle_cursor += mesh.triangle_count * 2
            if triangles.size == 0:
                instance_vertex_counts.append(0)
                continue

            used_vertices = np.unique(triangles.reshape(-1))
            compact_of_local = np.full(
                (mesh.vertex_count,), -1, dtype=np.int64
            )
            compact_of_local[used_vertices] = np.arange(
                used_vertices.size, dtype=np.int64
            )
            compact_triangles = compact_of_local[triangles]
            positions, normals = _world_geometry(scene_node, instance_index)
            compact_positions = positions[used_vertices]
            compact_normals = normals[used_vertices]
            compact_count = int(used_vertices.size)

            triangle_positions = compact_positions[compact_triangles]
            edge_01 = triangle_positions[:, 1] - triangle_positions[:, 0]
            edge_12 = triangle_positions[:, 2] - triangle_positions[:, 1]
            edge_20 = triangle_positions[:, 0] - triangle_positions[:, 2]
            edge_lengths = np.stack(
                (
                    np.linalg.norm(edge_01, axis=1),
                    np.linalg.norm(edge_12, axis=1),
                    np.linalg.norm(edge_20, axis=1),
                ),
                axis=1,
            )
            cross = np.cross(edge_01, -edge_20)
            cross_lengths = np.linalg.norm(cross, axis=1)
            denominator = np.sum(edge_lengths * edge_lengths, axis=1)
            qualities = np.divide(
                2.0 * math.sqrt(3.0) * cross_lengths,
                denominator,
                out=np.zeros_like(cross_lengths),
                where=denominator > 1e-30,
            )
            face_normals = np.divide(
                cross,
                cross_lengths[:, None],
                out=np.zeros_like(cross),
                where=cross_lengths[:, None] > 1e-30,
            )

            minimum_quality = np.ones((compact_count,), dtype=np.float64)
            minimum_edge = np.full(
                (compact_count,), np.inf, dtype=np.float64
            )
            maximum_edge = np.zeros((compact_count,), dtype=np.float64)
            flat_vertices = compact_triangles.reshape(-1)
            lumped_vertex_area = np.zeros(
                (compact_count,), dtype=np.float64
            )
            np.add.at(
                lumped_vertex_area,
                flat_vertices,
                np.repeat(0.5 * cross_lengths / 3.0, 3),
            )
            np.minimum.at(
                minimum_quality, flat_vertices, np.repeat(qualities, 3)
            )
            incident_minimum = np.stack(
                (
                    np.minimum(edge_lengths[:, 0], edge_lengths[:, 2]),
                    np.minimum(edge_lengths[:, 0], edge_lengths[:, 1]),
                    np.minimum(edge_lengths[:, 1], edge_lengths[:, 2]),
                ),
                axis=1,
            ).reshape(-1)
            incident_maximum = np.stack(
                (
                    np.maximum(edge_lengths[:, 0], edge_lengths[:, 2]),
                    np.maximum(edge_lengths[:, 0], edge_lengths[:, 1]),
                    np.maximum(edge_lengths[:, 1], edge_lengths[:, 2]),
                ),
                axis=1,
            ).reshape(-1)
            np.minimum.at(minimum_edge, flat_vertices, incident_minimum)
            np.maximum.at(maximum_edge, flat_vertices, incident_maximum)

            undirected_edges = np.concatenate(
                (
                    compact_triangles[:, (0, 1)],
                    compact_triangles[:, (1, 2)],
                    compact_triangles[:, (2, 0)],
                ),
                axis=0,
            )
            edge_a = np.minimum(
                undirected_edges[:, 0], undirected_edges[:, 1]
            )
            edge_b = np.maximum(
                undirected_edges[:, 0], undirected_edges[:, 1]
            )
            edge_keys = edge_a * compact_count + edge_b
            order = np.argsort(edge_keys, kind="stable")
            sorted_keys = edge_keys[order]
            sorted_faces = np.concatenate(
                (face_normals, face_normals, face_normals), axis=0
            )[order]
            _, unique_starts, unique_counts = np.unique(
                sorted_keys, return_index=True, return_counts=True
            )
            unique_edges = undirected_edges[order[unique_starts]]
            unique_a = np.minimum(unique_edges[:, 0], unique_edges[:, 1])
            unique_b = np.maximum(unique_edges[:, 0], unique_edges[:, 1])
            accepted = unique_counts <= 2
            paired = np.nonzero(unique_counts == 2)[0]
            if paired.size:
                first_normals = sorted_faces[unique_starts[paired]]
                second_normals = sorted_faces[unique_starts[paired] + 1]
                accepted[paired] = (
                    np.einsum(
                        "ij,ij->i", first_normals, second_normals
                    )
                    >= hard_edge_cosine
                )
            accepted &= unique_a != unique_b
            unique_a = unique_a[accepted]
            unique_b = unique_b[accepted]
            normal_cosines = np.maximum(
                np.einsum(
                    "ij,ij->i",
                    compact_normals[unique_a],
                    compact_normals[unique_b],
                ),
                0.0,
            )
            edge_weights = normal_cosines**4
            nonzero_weights = edge_weights > 1e-4
            unique_a = unique_a[nonzero_weights]
            unique_b = unique_b[nonzero_weights]
            edge_weights = edge_weights[nonzero_weights].astype(np.float32)

            # glTF commonly duplicates render vertices at UV seams. Reconnect
            # geometric copies when their normals remain compatible so the
            # solve follows logical surface topology instead of UV topology.
            extent = np.ptp(compact_positions, axis=0)
            weld_tolerance = max(float(np.linalg.norm(extent)) * 1e-7, 1e-8)
            quantized_positions = np.rint(
                compact_positions / weld_tolerance
            ).astype(np.int64)
            _, weld_groups = np.unique(
                quantized_positions, axis=0, return_inverse=True
            )
            weld_order = np.argsort(weld_groups, kind="stable")
            sorted_groups = weld_groups[weld_order]
            first_in_group = np.empty((compact_count,), dtype=np.bool_)
            first_in_group[0] = True
            first_in_group[1:] = sorted_groups[1:] != sorted_groups[:-1]
            group_start = np.maximum.accumulate(
                np.where(first_in_group, np.arange(compact_count), 0)
            )
            weld_members = weld_order[~first_in_group]
            weld_representatives = weld_order[group_start[~first_in_group]]
            if weld_members.size:
                weld_normal_cosines = np.einsum(
                    "ij,ij->i",
                    compact_normals[weld_members],
                    compact_normals[weld_representatives],
                )
                compatible = weld_normal_cosines >= hard_edge_cosine
                weld_members = weld_members[compatible]
                weld_representatives = weld_representatives[compatible]
                weld_weights = np.maximum(
                    weld_normal_cosines[compatible], 0.0
                ).astype(np.float32) ** 4
                unique_a = np.concatenate(
                    (unique_a, weld_representatives)
                )
                unique_b = np.concatenate((unique_b, weld_members))
                edge_weights = np.concatenate(
                    (edge_weights, weld_weights)
                )
            instance_weld_edge_count = int(weld_members.size)

            finite_minimum = np.where(
                np.isfinite(minimum_edge), minimum_edge, maximum_edge
            )
            scale_ratio = finite_minimum / np.maximum(maximum_edge, 1e-30)
            ill_condition = np.maximum(
                1.0 - np.clip(minimum_quality, 0.0, 1.0),
                1.0 - np.sqrt(np.clip(scale_ratio, 0.0, 1.0)),
            )
            conditions = 1.0 + (
                maximum_condition_multiplier - 1.0
            ) * ill_condition**2
            if probe_layout is not None:
                # Triangle shape is only half of vertex-lighting
                # conditioning. A perfectly shaped but enormous triangle is
                # still badly under-resolved when one vertex coefficient
                # represents hundreds of surface-probe kernels.
                kernel_radius = max(
                    float(
                        probe_layout.instance_gpu_data[instance_index][
                            "params"
                        ][0]
                    ),
                    1e-8,
                )
                footprint_ratio = (
                    np.sqrt(np.maximum(lumped_vertex_area, 0.0))
                    / kernel_radius
                )
                footprint_ill_condition = np.clip(
                    np.log2(np.maximum(footprint_ratio, 1.0)) / 8.0,
                    0.0,
                    1.0,
                )
                footprint_conditions = 1.0 + (
                    maximum_condition_multiplier - 1.0
                ) * footprint_ill_condition**2
                conditions = np.maximum(conditions, footprint_conditions)
            condition_arrays.append(conditions)

            double_sided = bool(
                scene_node.materials[material_id].double_sided
            )
            side_count = 2 if double_sided else 1
            weld_edge_count += instance_weld_edge_count * side_count
            side_bases: list[int] = []
            for side in range(side_count):
                side_base = vertex_cursor
                vertex_cursor += compact_count
                side_bases.append(side_base)
                meta = np.empty((compact_count, 4), dtype=np.uint32)
                meta[:, 0] = instance_index
                meta[:, 1] = used_vertices.astype(np.uint32)
                meta[:, 2] = side
                meta[:, 3] = conditions.astype(np.float32).view(np.uint32)
                vertex_meta_arrays.append(meta)
                position_area = np.empty(
                    (compact_count, 4), dtype=np.float32
                )
                position_area[:, :3] = compact_positions.astype(np.float32)
                position_area[:, 3] = lumped_vertex_area.astype(np.float32)
                vertex_position_area_arrays.append(position_area)
                normal = np.zeros((compact_count, 4), dtype=np.float32)
                normal[:, :3] = compact_normals.astype(np.float32) * (
                    -1.0 if side else 1.0
                )
                vertex_normal_arrays.append(normal)
                neighbor_source_arrays.append(
                    side_base + np.concatenate((unique_a, unique_b))
                )
                neighbor_destination_arrays.append(
                    side_base + np.concatenate((unique_b, unique_a))
                )
                neighbor_weight_arrays.append(
                    np.concatenate((edge_weights, edge_weights))
                )

            if probe_layout is not None:
                instance_gpu_data = probe_layout.instance_gpu_data[
                    instance_index
                ]
                probe_offset = int(instance_gpu_data["offsets"][2])
                probe_count = int(instance_gpu_data["offsets"][3])
                instance_probes = probe_layout.probes[
                    probe_offset : probe_offset + probe_count
                ]
                if instance_probes.size:
                    flags = instance_probes["meta"][:, 3]
                    reconstruction_mask = (
                        flags & np.uint32(SURFACE_PROBE_FLAG_VERTEX_ANCHOR)
                    ) == 0
                    local_probe_indices = np.flatnonzero(
                        reconstruction_mask
                    )
                    projection_probes = instance_probes[
                        reconstruction_mask
                    ]
                    triangle_ids = projection_probes["meta"][
                        :, 0
                    ].astype(np.int64)
                    side_ids = (
                        (
                            projection_probes["meta"][:, 3]
                            & np.uint32(SURFACE_PROBE_FLAG_BACK_SIDE)
                        )
                        != 0
                    ).astype(np.int64)
                    valid = (
                        (triangle_ids >= 0)
                        & (triangle_ids < mesh.triangle_count)
                        & (side_ids < side_count)
                    )
                    if np.any(valid):
                        triangle_ids = triangle_ids[valid]
                        side_ids = side_ids[valid]
                        projection_probes = projection_probes[valid]
                        global_probe_indices = (
                            probe_offset + local_probe_indices[valid]
                        ).astype(np.uint32)
                        barycentrics = np.stack(
                            (
                                1.0
                                - projection_probes["meta"][:, 1].view(
                                    np.float32
                                )
                                - projection_probes["meta"][:, 2].view(
                                    np.float32
                                ),
                                projection_probes["meta"][:, 1].view(
                                    np.float32
                                ),
                                projection_probes["meta"][:, 2].view(
                                    np.float32
                                ),
                            ),
                            axis=1,
                        ).astype(np.float64)
                        barycentrics = np.maximum(barycentrics, 0.0)
                        barycentrics /= np.maximum(
                            np.sum(barycentrics, axis=1, keepdims=True),
                            1e-30,
                        )

                        # Adaptive WSE places approximately m(x)=1/f(x)
                        # samples per unit area. Weighting each sample by f(x)
                        # cancels that density, then normalizing per
                        # triangle/side makes repair samples unable to bias the
                        # area integral merely because they are extra points.
                        support_f = np.clip(
                            projection_probes["normal_side"][:, 3].astype(
                                np.float64
                            ),
                            1e-4,
                            1.0,
                        )
                        group_ids = triangle_ids * side_count + side_ids
                        group_support = np.bincount(
                            group_ids,
                            weights=support_f,
                            minlength=mesh.triangle_count * side_count,
                        )
                        triangle_areas = 0.5 * cross_lengths
                        sample_area = (
                            triangle_areas[triangle_ids]
                            * support_f
                            / np.maximum(group_support[group_ids], 1e-30)
                        )
                        projection_weights = (
                            sample_area[:, None] * barycentrics
                        ).reshape(-1)
                        projection_coverage_weights = np.repeat(
                            (
                                triangle_areas[triangle_ids]
                                * support_f
                                / np.maximum(
                                    group_support[group_ids], 1e-30
                                )
                                / 3.0
                            )[:, None],
                            3,
                            axis=1,
                        ).reshape(-1)
                        projection_vertices = np.empty(
                            (triangle_ids.size, 3), dtype=np.int64
                        )
                        for side, side_base in enumerate(side_bases):
                            side_mask = side_ids == side
                            projection_vertices[side_mask] = (
                                side_base
                                + compact_triangles[
                                    triangle_ids[side_mask]
                                ]
                            )
                        projection_sources = projection_vertices.reshape(-1)
                        projection_probe_indices = np.repeat(
                            global_probe_indices, 3
                        )
                        positive = projection_coverage_weights > 1e-20
                        projection_source_arrays.append(
                            projection_sources[positive]
                        )
                        projection_probe_arrays.append(
                            projection_probe_indices[positive]
                        )
                        projection_weight_arrays.append(
                            projection_weights[positive].astype(np.float32)
                        )
                        projection_coverage_weight_arrays.append(
                            projection_coverage_weights[positive].astype(
                                np.float32
                            )
                        )

            triangle_records = np.zeros(
                (mesh.triangle_count * 2, 4), dtype=np.uint32
            )
            triangle_records[0::2, :3] = (
                side_bases[0] + compact_triangles
            ).astype(np.uint32)
            triangle_records[0::2, 3] = 1
            if double_sided:
                triangle_records[1::2, :3] = (
                    side_bases[1] + compact_triangles
                ).astype(np.uint32)
                triangle_records[1::2, 3] = 1
            triangle_record_arrays.append(triangle_records)
            instance_vertex_counts.append(compact_count * side_count)

        meta_array = (
            np.concatenate(vertex_meta_arrays, axis=0)
            if vertex_meta_arrays
            else np.zeros((0, 4), dtype=np.uint32)
        )
        position_area_array = (
            np.concatenate(vertex_position_area_arrays, axis=0)
            if vertex_position_area_arrays
            else np.zeros((0, 4), dtype=np.float32)
        )
        normal_array = (
            np.concatenate(vertex_normal_arrays, axis=0)
            if vertex_normal_arrays
            else np.zeros((0, 4), dtype=np.float32)
        )
        if neighbor_source_arrays:
            sources = np.concatenate(neighbor_source_arrays).astype(np.int64)
            destinations = np.concatenate(
                neighbor_destination_arrays
            ).astype(np.uint32)
            weights = np.concatenate(neighbor_weight_arrays).astype(np.float32)
            neighbor_order = np.lexsort((destinations, sources))
            sources = sources[neighbor_order]
            destinations = destinations[neighbor_order]
            weights = weights[neighbor_order]
            counts = np.bincount(sources, minlength=vertex_cursor)
            offsets = np.empty((vertex_cursor + 1,), dtype=np.uint32)
            offsets[0] = 0
            offsets[1:] = np.cumsum(counts, dtype=np.uint32)
            neighbor_array = np.stack(
                (destinations, weights.view(np.uint32)), axis=1
            )
        else:
            offsets = np.zeros((vertex_cursor + 1,), dtype=np.uint32)
            neighbor_array = np.zeros((0, 2), dtype=np.uint32)
        if projection_source_arrays:
            projection_sources = np.concatenate(
                projection_source_arrays
            ).astype(np.int64)
            projection_probes = np.concatenate(
                projection_probe_arrays
            ).astype(np.uint32)
            projection_weights = np.concatenate(
                projection_weight_arrays
            ).astype(np.float32)
            projection_coverage_weights = np.concatenate(
                projection_coverage_weight_arrays
            ).astype(np.float32)
            projection_order = np.lexsort(
                (projection_probes, projection_sources)
            )
            projection_sources = projection_sources[projection_order]
            projection_probes = projection_probes[projection_order]
            projection_weights = projection_weights[projection_order]
            projection_coverage_weights = projection_coverage_weights[
                projection_order
            ]
            projection_counts = np.bincount(
                projection_sources, minlength=vertex_cursor
            )
            projection_offsets = np.empty(
                (vertex_cursor + 1,), dtype=np.uint32
            )
            projection_offsets[0] = 0
            projection_offsets[1:] = np.cumsum(
                projection_counts, dtype=np.uint32
            )
            projection_array = np.stack(
                (
                    projection_probes,
                    projection_weights.view(np.uint32),
                    projection_coverage_weights.view(np.uint32),
                ),
                axis=1,
            )
        else:
            projection_offsets = np.zeros(
                (vertex_cursor + 1,), dtype=np.uint32
            )
            projection_array = np.zeros((0, 3), dtype=np.uint32)
        projection_mass = np.zeros((vertex_cursor,), dtype=np.float64)
        if projection_source_arrays:
            np.add.at(
                projection_mass,
                projection_sources,
                projection_coverage_weights.astype(np.float64),
            )
        expected_mass = position_area_array[:, 3].astype(np.float64)
        projection_coverage = np.divide(
            projection_mass,
            expected_mass,
            out=np.zeros_like(projection_mass),
            where=expected_mass > 1e-30,
        )
        zero_projection_vertex_count = int(
            np.count_nonzero(projection_coverage <= 1e-6)
        )
        partial_projection_vertex_count = int(
            np.count_nonzero(
                (projection_coverage > 1e-6)
                & (projection_coverage < 0.999)
            )
        )
        triangle_array = (
            np.concatenate(triangle_record_arrays, axis=0)
            if triangle_record_arrays
            else np.zeros((0, 4), dtype=np.uint32)
        )
        instance_array = np.asarray(
            instance_offsets, dtype=np.uint32
        ).reshape((-1, 2))
        conditions_array = (
            np.concatenate(condition_arrays)
            if condition_arrays
            else np.ones((0,), dtype=np.float64)
        )
        return cls(
            vertex_meta=meta_array,
            vertex_position_area=position_area_array,
            vertex_normal=normal_array,
            neighbor_offsets=offsets,
            neighbors=neighbor_array,
            projection_offsets=projection_offsets,
            projection_samples=projection_array,
            triangle_map=triangle_array,
            instance_offsets=instance_array,
            instance_vertex_counts=tuple(instance_vertex_counts),
            weld_edge_count=weld_edge_count,
            condition_mean=(
                float(np.mean(conditions_array))
                if conditions_array.size
                else 1.0
            ),
            condition_p95=(
                float(np.percentile(conditions_array, 95.0))
                if conditions_array.size
                else 1.0
            ),
            zero_projection_vertex_count=zero_projection_vertex_count,
            partial_projection_vertex_count=(
                partial_projection_vertex_count
            ),
        )


class SurfaceProbeVertexLighting:
    def __init__(
        self,
        device: spy.Device,
        scene: Scene,
        path_tracer: SurfaceProbePathTracer,
        *,
        profile_sink: list[tuple[str, float]] | None = None,
    ):
        self.device = device
        self.scene = scene
        self.path_tracer = path_tracer
        start = time.perf_counter()
        self.layout = VertexLightingLayout.build(
            scene.scene_node,
            probe_layout=path_tracer.layout,
        )
        if profile_sink is not None:
            profile_sink.append(
                ("vertex_lighting_layout", time.perf_counter() - start)
            )

        self.gather_pipeline = device.create_compute_pipeline(
            device.load_program(
                "surface_probe_vertex_lighting.slang", ["gather_main"]
            )
        )
        self.smooth_pipeline = device.create_compute_pipeline(
            device.load_program(
                "surface_probe_vertex_lighting.slang", ["smooth_main"]
            )
        )
        self.pack_pipeline = device.create_compute_pipeline(
            device.load_program(
                "surface_probe_vertex_lighting.slang", ["pack_main"]
            )
        )
        self.vertex_meta_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_meta",
            data=np.ascontiguousarray(self.layout.vertex_meta),
        )
        self.vertex_position_area_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_position_area",
            data=np.ascontiguousarray(self.layout.vertex_position_area),
        )
        self.vertex_normal_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_normal",
            data=np.ascontiguousarray(self.layout.vertex_normal),
        )
        self.neighbor_offsets_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_neighbor_offsets",
            data=np.ascontiguousarray(self.layout.neighbor_offsets),
        )
        neighbor_data = (
            self.layout.neighbors
            if self.layout.neighbors.size
            else np.zeros((1, 2), dtype=np.uint32)
        )
        self.neighbors_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_neighbors",
            data=np.ascontiguousarray(neighbor_data),
        )
        self.projection_offsets_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_projection_offsets",
            data=np.ascontiguousarray(self.layout.projection_offsets),
        )
        projection_data = (
            self.layout.projection_samples
            if self.layout.projection_samples.size
            else np.zeros((1, 3), dtype=np.uint32)
        )
        self.projection_samples_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_projection_samples",
            data=np.ascontiguousarray(projection_data),
        )
        self.triangle_map_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_triangle_map",
            data=np.ascontiguousarray(self.layout.triangle_map),
        )
        self.instance_offsets_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_lighting_instance_offsets",
            data=np.ascontiguousarray(self.layout.instance_offsets),
        )
        self.built = False
        self.last_pass_count = 0
        self.last_regularization = 0.0
        self.last_rgbm_range = 0.0

    def _buffer(
        self,
        render_data: RenderData,
        key: str,
        *,
        stride: int,
    ) -> spy.Buffer:
        return render_data.get_buffer(
            key,
            usage=spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource,
            struct_size=stride,
            element_count=max(self.layout.vertex_count, 1),
            label=key.rsplit(".", 1)[-1],
        )

    def packed_buffer(self, render_data: RenderData) -> spy.Buffer:
        return self._buffer(
            render_data, VERTEX_LIGHTING_RGBM_KEY, stride=4
        )

    def target_buffer(self, render_data: RenderData) -> spy.Buffer:
        return self._buffer(
            render_data, VERTEX_LIGHTING_TARGET_KEY, stride=16
        )

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        render_data: RenderData,
        probe_irradiance: spy.Buffer,
        *,
        min_gather_count: int,
        smoothing_passes: int,
        regularization_strength: float,
        rgbm_range: float,
    ) -> spy.Buffer:
        target = self.target_buffer(render_data)
        work_a = self._buffer(
            render_data, VERTEX_LIGHTING_WORK_A_KEY, stride=16
        )
        work_b = self._buffer(
            render_data, VERTEX_LIGHTING_WORK_B_KEY, stride=16
        )
        packed = self.packed_buffer(render_data)
        count = self.layout.vertex_count
        if count == 0:
            self.built = True
            return packed

        with command_encoder.begin_compute_pass() as pass_encoder:
            shader = pass_encoder.bind_pipeline(self.gather_pipeline)
            cursor = spy.ShaderCursor(shader)
            cursor.g_probe_irradiance = probe_irradiance
            cursor.g_surface_probes = self.path_tracer.probe_buffer
            cursor.g_surface_probe_nodes = self.path_tracer.node_buffer
            cursor.g_surface_probe_instances = (
                self.path_tracer.instance_buffer
            )
            cursor.g_projection_offsets = self.projection_offsets_buffer
            cursor.g_projection_samples = self.projection_samples_buffer
            cursor.g_vertex_meta = self.vertex_meta_buffer
            cursor.g_vertex_position_area = (
                self.vertex_position_area_buffer
            )
            cursor.g_vertex_normal = self.vertex_normal_buffer
            cursor.g_target = target
            cursor.g_destination = work_a
            cursor.g_vertex_count = count
            cursor.g_min_gather_count = max(
                1, min(32, int(min_gather_count))
            )
            pass_encoder.dispatch(thread_count=[count, 1, 1])

        current = work_a
        smoothing_passes = max(0, int(smoothing_passes))
        for pass_index in range(smoothing_passes):
            destination = work_b if current is work_a else work_a
            with command_encoder.begin_compute_pass() as pass_encoder:
                shader = pass_encoder.bind_pipeline(self.smooth_pipeline)
                cursor = spy.ShaderCursor(shader)
                cursor.g_vertex_meta = self.vertex_meta_buffer
                cursor.g_neighbor_offsets = self.neighbor_offsets_buffer
                cursor.g_neighbors = self.neighbors_buffer
                cursor.g_target = target
                cursor.g_source = current
                cursor.g_destination = destination
                cursor.g_vertex_count = count
                cursor.g_regularization_strength = max(
                    0.0, float(regularization_strength)
                )
                pass_encoder.dispatch(thread_count=[count, 1, 1])
            current = destination

        with command_encoder.begin_compute_pass() as pass_encoder:
            shader = pass_encoder.bind_pipeline(self.pack_pipeline)
            cursor = spy.ShaderCursor(shader)
            cursor.g_source = current
            cursor.g_packed_rgbm = packed
            cursor.g_vertex_count = count
            cursor.g_rgbm_range = max(1e-3, float(rgbm_range))
            pass_encoder.dispatch(thread_count=[count, 1, 1])

        self.built = True
        self.last_pass_count = smoothing_passes
        self.last_regularization = max(
            0.0, float(regularization_strength)
        )
        self.last_rgbm_range = max(1e-3, float(rgbm_range))
        return packed
