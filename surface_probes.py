from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from itertools import product
from typing import Literal
import warnings

import numpy as np
import numpy.typing as npt
import slangpy as spy

from scene_node import SceneNode


SURFACE_PROBE_PAYLOAD_SIZE = 16
SURFACE_PROBE_METADATA_SIZE = 48
SURFACE_PROBE_NODE_SIZE = 16
SURFACE_PROBE_INSTANCE_SIZE = 48
SURFACE_PROBE_FLAG_BACK_SIDE = 1 << 0
SURFACE_PROBE_FLAG_VERTEX_ANCHOR = 1 << 1
SURFACE_PROBE_FLAG_PROTECTED = 1 << 2

SurfaceProbeSamplerBackend = Literal["auto", "cpp", "python"]
_cpp_sampler_fallback_warning_printed = False

_NEIGHBOR_CELL_OFFSETS = tuple(product((-1, 0, 1), repeat=3))

SURFACE_PROBE_DTYPE = np.dtype(
    [
        ("position_radius", np.float32, (4,)),
        ("normal_side", np.float32, (4,)),
        ("meta", np.uint32, (4,)),
    ],
    align=False,
)

SURFACE_PROBE_INSTANCE_DTYPE = np.dtype(
    [
        ("root_center_extent", np.float32, (4,)),
        ("offsets", np.uint32, (4,)),
        ("params", np.float32, (4,)),
    ],
    align=False,
)


@dataclass(frozen=True)
class SurfaceProbeInstanceInfo:
    root_center: tuple[float, float, float]
    root_extent: float
    node_offset: int
    node_count: int
    probe_offset: int
    probe_count: int
    kernel_radius: float
    normal_cosine_threshold: float
    plane_sigma: float
    surface_area: float
    base_surface_site_count: int
    repair_surface_site_count: int
    protected_surface_site_count: int
    surface_site_count: int
    candidate_count: int
    audit_point_count: int
    zero_gather_before: int
    zero_gather_after_repair: int
    zero_gather_after: int
    deficit_point_count_before: int
    deficit_point_count_after_repair: int
    deficit_point_count_after: int
    ess_p50_before: float
    ess_p50_after: float
    support_f_p10: float
    support_f_p50: float
    density_m_p95: float
    adaptive_density_mean: float
    adaptive_density_p95: float
    reconstruction_probe_count: int
    vertex_anchor_site_count: int
    vertex_anchor_probe_count: int


@dataclass(frozen=True)
class _InstanceCandidates:
    positions: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    barycentrics: npt.NDArray[np.float32]
    triangle_indices: npt.NDArray[np.uint32]
    surface_area: float


@dataclass
class _InstanceBuildState:
    instance_index: int
    base_candidates: _InstanceCandidates
    base_selected: npt.NDArray[np.int64]
    audit_candidates: _InstanceCandidates
    repair_candidates: _InstanceCandidates
    support_candidates: _InstanceCandidates
    support_area_weights: npt.NDArray[np.float32]
    candidate_relative_densities: npt.NDArray[np.float32]
    adaptive_density_mean: float
    adaptive_density_p95: float
    kernel_radius: float
    poisson_radius: float
    target_count: int
    double_sided: bool
    vertex_anchors: _InstanceCandidates
    triangle_vertex_anchor_indices: npt.NDArray[np.uint32]


@dataclass
class _AdaptiveInstancePrepass:
    instance_index: int
    triangles: npt.NDArray[np.float32]
    triangle_indices: npt.NDArray[np.uint32]
    mesh_triangle_indices: npt.NDArray[np.uint32]
    triangle_areas: npt.NDArray[np.float64]
    surface_area: float
    triangle_densities: npt.NDArray[np.float32]
    adaptive_mass: float


class _PointGrid:
    def __init__(
        self,
        positions: npt.NDArray[np.float32],
        cell_size: float,
        active: npt.NDArray[np.bool_] | None = None,
    ):
        self.positions = positions
        self.cell_size = max(float(cell_size), 1e-8)
        self.inverse_cell_size = 1.0 / self.cell_size
        self.active = active
        self.cells: dict[tuple[int, int, int], list[int]] = {}
        coordinates = np.floor(
            positions.astype(np.float64) * self.inverse_cell_size
        ).astype(np.int64)
        for index, coordinate in enumerate(coordinates):
            if active is not None and not bool(active[index]):
                continue
            key = (int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))
            self.cells.setdefault(key, []).append(index)

    def nearby_indices(self, position: npt.NDArray[np.float32]) -> list[int]:
        coordinate = np.floor(
            position.astype(np.float64) * self.inverse_cell_size
        ).astype(np.int64)
        result: list[int] = []
        for offset in _NEIGHBOR_CELL_OFFSETS:
            key = (
                int(coordinate[0]) + offset[0],
                int(coordinate[1]) + offset[1],
                int(coordinate[2]) + offset[2],
            )
            for index in self.cells.get(key, ()):
                if self.active is None or bool(self.active[index]):
                    result.append(index)
        return result


def _allocate_counts(
    weights: npt.NDArray[np.float64],
    total: int,
    *,
    minimum_per_entry: int = 1,
) -> np.ndarray:
    count = int(weights.shape[0])
    total = int(total)
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    minimum_per_entry = max(1, int(minimum_per_entry))
    if total < count * minimum_per_entry:
        raise ValueError(
            f"Surface probe budget {total} is smaller than the required "
            f"{minimum_per_entry} samples for each of {count} entries"
        )
    result = np.full((count,), minimum_per_entry, dtype=np.int64)
    remaining = total - count * minimum_per_entry
    if remaining == 0:
        return result
    normalized = weights / max(float(np.sum(weights)), 1e-30)
    expected = normalized * float(remaining)
    extra = np.floor(expected).astype(np.int64)
    result += extra
    remainder = remaining - int(np.sum(extra))
    if remainder > 0:
        order = np.argsort(-(expected - extra), kind="stable")
        result[order[:remainder]] += 1
    return result


def _transform_positions(
    matrix: spy.float4x4,
    positions: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    transform = np.asarray(matrix.to_numpy(), dtype=np.float64)
    return np.asarray(
        positions.astype(np.float64) @ transform[:3, :3].T
        + transform[:3, 3],
        dtype=np.float32,
    )


def _instance_triangles(
    scene_node: SceneNode,
    instance_index: int,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    float,
]:
    mesh_id, _, transform_id = scene_node.instances[instance_index]
    mesh = scene_node.meshes[mesh_id]
    world_vertices = _transform_positions(
        scene_node.transforms[transform_id].matrix,
        mesh.vertices[:, 0:3],
    )
    triangles = world_vertices[mesh.indices.astype(np.int64)]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edges_a, edges_b)
    doubled_areas = np.linalg.norm(cross, axis=1)
    valid = doubled_areas > 1e-12
    if not np.any(valid):
        return (
            np.zeros((0, 3, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0, 3), dtype=np.uint32),
            0.0,
        )
    return (
        triangles[valid].astype(np.float32),
        np.nonzero(valid)[0].astype(np.uint32),
        mesh.indices[valid].astype(np.uint32),
        float(0.5 * np.sum(doubled_areas[valid], dtype=np.float64)),
    )


def _instance_vertex_anchors(
    scene_node: SceneNode,
    instance_index: int,
    triangle_indices: npt.NDArray[np.uint32],
    mesh_triangle_indices: npt.NDArray[np.uint32],
    surface_area: float,
) -> tuple[_InstanceCandidates, npt.NDArray[np.uint32]]:
    mesh_id, _, transform_id = scene_node.instances[instance_index]
    mesh = scene_node.meshes[mesh_id]
    vertex_indices = np.unique(mesh_triangle_indices.reshape(-1)).astype(
        np.uint32
    )


    vertex_to_anchor = np.full(
        (mesh.vertex_count,), np.iinfo(np.uint32).max, dtype=np.uint32
    )
    vertex_to_anchor[vertex_indices] = np.arange(
        vertex_indices.shape[0], dtype=np.uint32
    )

    transform = np.asarray(
        scene_node.transforms[transform_id].matrix.to_numpy(), dtype=np.float64
    )
    positions = _transform_positions(
        scene_node.transforms[transform_id].matrix,
        mesh.vertices[vertex_indices, :3],
    )
    normal_matrix = np.linalg.inv(transform[:3, :3]).T
    normals = np.asarray(
        mesh.vertices[vertex_indices, 3:6].astype(np.float64)
        @ normal_matrix.T,
        dtype=np.float32,
    )
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(normal_lengths, 1e-20)

    flat_vertices = mesh_triangle_indices.reshape(-1)
    flat_triangles = np.repeat(triangle_indices, 3)
    flat_corners = np.tile(np.arange(3, dtype=np.int64), triangle_indices.size)
    first_incident: dict[int, tuple[int, int]] = {}
    for vertex, triangle, corner in zip(
        flat_vertices, flat_triangles, flat_corners
    ):
        first_incident.setdefault(int(vertex), (int(triangle), int(corner)))
    anchor_triangles = np.empty((vertex_indices.shape[0],), dtype=np.uint32)
    barycentrics = np.zeros((vertex_indices.shape[0], 3), dtype=np.float32)
    for anchor, vertex in enumerate(vertex_indices):
        triangle, corner = first_incident[int(vertex)]
        anchor_triangles[anchor] = triangle
        barycentrics[anchor, corner] = 1.0

    triangle_map = np.full(
        (mesh.triangle_count, 3),
        np.iinfo(np.uint32).max,
        dtype=np.uint32,
    )
    triangle_map[triangle_indices] = vertex_to_anchor[mesh_triangle_indices]
    return (
        _InstanceCandidates(
            positions=positions,
            normals=normals,
            barycentrics=barycentrics,
            triangle_indices=anchor_triangles,
            surface_area=float(surface_area),
        ),
        triangle_map,
    )


def _sample_instance_surface(
    triangles: npt.NDArray[np.float32],
    triangle_indices: npt.NDArray[np.uint32],
    sample_count: int,
    seed: int,
    triangle_densities: npt.ArrayLike | None = None,
) -> _InstanceCandidates:
    sample_count = max(1, int(sample_count))
    p0 = triangles[:, 0]
    edge_a = triangles[:, 1] - p0
    edge_b = triangles[:, 2] - p0
    cross = np.cross(edge_a, edge_b)
    doubled_areas = np.linalg.norm(cross, axis=1)
    area = 0.5 * doubled_areas
    total_area = float(np.sum(area, dtype=np.float64))
    if triangle_densities is None:
        sampling_weights = area
    else:
        densities = np.asarray(triangle_densities, dtype=np.float64)
        if densities.shape != area.shape or not np.all(
            np.isfinite(densities) & (densities > 0.0)
        ):
            raise ValueError(
                "triangle_densities must be finite, positive, and match triangles"
            )
        sampling_weights = area * densities
    cdf = np.cumsum(sampling_weights, dtype=np.float64)
    cdf /= cdf[-1]
    cdf[-1] = 1.0

    rng = np.random.default_rng(int(seed))
    choices = np.searchsorted(cdf, rng.random(sample_count), side="right")
    choices = np.minimum(choices, triangles.shape[0] - 1)
    random_uv = rng.random((sample_count, 2), dtype=np.float32)
    sqrt_u = np.sqrt(random_uv[:, 0])
    b0 = 1.0 - sqrt_u
    b1 = sqrt_u * (1.0 - random_uv[:, 1])
    b2 = sqrt_u * random_uv[:, 1]
    barycentrics = np.stack((b0, b1, b2), axis=1).astype(np.float32)
    positions = (
        p0[choices]
        + edge_a[choices] * b1[:, None]
        + edge_b[choices] * b2[:, None]
    ).astype(np.float32)
    face_normals = cross / np.maximum(doubled_areas[:, None], 1e-30)
    normals = face_normals[choices].astype(np.float32)
    return _InstanceCandidates(
        positions=positions,
        normals=normals,
        barycentrics=barycentrics,
        triangle_indices=triangle_indices[choices],
        surface_area=total_area,
    )


def _subset_candidates(
    candidates: _InstanceCandidates,
    indices: npt.ArrayLike,
) -> _InstanceCandidates:
    selected = np.asarray(indices, dtype=np.int64)
    return _InstanceCandidates(
        positions=candidates.positions[selected].copy(),
        normals=candidates.normals[selected].copy(),
        barycentrics=candidates.barycentrics[selected].copy(),
        triangle_indices=candidates.triangle_indices[selected].copy(),
        surface_area=candidates.surface_area,
    )


def _repeat_candidate_subset_by_count(
    candidates: _InstanceCandidates,
    indices: npt.ArrayLike,
    repeat_counts: npt.ArrayLike,
) -> _InstanceCandidates:
    """Repeat face-local candidates by their exact residual deficits.

    Repeated sites deliberately share a position. They are independent probe
    records with independent progressive samples, and the common position
    guarantees strong, visible support at the audit point that created them.
    """

    selected = np.asarray(indices, dtype=np.int64)
    counts = np.asarray(repeat_counts, dtype=np.int64)
    if counts.shape != selected.shape:
        raise ValueError("repeat_counts must match the selected indices")
    if np.any(counts < 0):
        raise ValueError("repeat_counts must be non-negative")
    if selected.size == 0 or not np.any(counts):
        return _combine_candidate_sets([], candidates.surface_area)
    repeated = np.repeat(selected, counts)
    return _subset_candidates(candidates, repeated)


def _combine_candidate_sets(
    candidates: list[_InstanceCandidates],
    surface_area: float,
) -> _InstanceCandidates:
    non_empty = [item for item in candidates if item.positions.shape[0] > 0]
    if not non_empty:
        return _InstanceCandidates(
            positions=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            barycentrics=np.zeros((0, 3), dtype=np.float32),
            triangle_indices=np.zeros((0,), dtype=np.uint32),
            surface_area=float(surface_area),
        )
    return _InstanceCandidates(
        positions=np.concatenate([item.positions for item in non_empty]),
        normals=np.concatenate([item.normals for item in non_empty]),
        barycentrics=np.concatenate(
            [item.barycentrics for item in non_empty]
        ),
        triangle_indices=np.concatenate(
            [item.triangle_indices for item in non_empty]
        ),
        surface_area=float(surface_area),
    )


def _triangle_centroid_candidates(
    triangles: npt.NDArray[np.float32],
    triangle_indices: npt.NDArray[np.uint32],
    surface_area: float,
) -> _InstanceCandidates:
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(lengths[:, None], 1e-30)
    return _InstanceCandidates(
        positions=np.mean(triangles, axis=1, dtype=np.float32),
        normals=normals.astype(np.float32),
        barycentrics=np.full(
            (triangles.shape[0], 3), 1.0 / 3.0, dtype=np.float32
        ),
        triangle_indices=triangle_indices.copy(),
        surface_area=float(surface_area),
    )


def _triangle_inset_corner_candidates(
    triangles: npt.NDArray[np.float32],
    triangle_indices: npt.NDArray[np.uint32],
    inset_distance: float,
    surface_area: float,
) -> _InstanceCandidates:
    """Create face-local corner audits without sitting exactly on a seam."""

    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    maximum_edges = np.maximum(np.max(edge_lengths, axis=1), 1e-8)
    inset = np.clip(
        float(inset_distance) / maximum_edges,
        1e-4,
        0.1,
    ).astype(np.float32)
    barycentrics = np.repeat(inset[:, None], 9, axis=1).reshape(-1, 3)
    for corner in range(3):
        barycentrics[corner::3, corner] = 1.0 - 2.0 * inset
    repeated_triangles = np.repeat(triangles, 3, axis=0)
    positions = np.einsum(
        "ni,nij->nj", barycentrics, repeated_triangles
    ).astype(np.float32)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_a, edge_b)
    face_normals = cross / np.maximum(
        np.linalg.norm(cross, axis=1)[:, None], 1e-30
    )
    return _InstanceCandidates(
        positions=positions,
        normals=np.repeat(face_normals.astype(np.float32), 3, axis=0),
        barycentrics=barycentrics,
        triangle_indices=np.repeat(triangle_indices, 3),
        surface_area=float(surface_area),
    )


def _boundary_edge_candidates(
    triangles: npt.NDArray[np.float32],
    triangle_indices: npt.NDArray[np.uint32],
    mesh_triangle_indices: npt.NDArray[np.uint32],
    spacing: float,
    surface_area: float,
) -> _InstanceCandidates:
    local_edges = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    edge_vertices = mesh_triangle_indices[:, local_edges].reshape(-1, 2)
    edge_keys = np.sort(edge_vertices, axis=1)
    _, first, counts = np.unique(
        edge_keys, axis=0, return_index=True, return_counts=True
    )
    boundary_flat = np.sort(first[counts == 1])
    if boundary_flat.size == 0:
        return _combine_candidate_sets([], surface_area)

    face_indices = boundary_flat // 3
    local_edge_indices = boundary_flat % 3
    edge_starts = local_edges[local_edge_indices, 0]
    edge_ends = local_edges[local_edge_indices, 1]
    start_positions = triangles[face_indices, edge_starts]
    end_positions = triangles[face_indices, edge_ends]
    lengths = np.linalg.norm(end_positions - start_positions, axis=1)
    segment_counts = np.maximum(
        np.ceil(lengths / max(float(spacing), 1e-8)).astype(np.int64), 1
    )
    total = int(np.sum(segment_counts, dtype=np.int64))
    positions = np.empty((total, 3), dtype=np.float32)
    normals = np.empty((total, 3), dtype=np.float32)
    barycentrics = np.zeros((total, 3), dtype=np.float32)
    output_triangles = np.empty((total,), dtype=np.uint32)
    face_edges_a = triangles[:, 1] - triangles[:, 0]
    face_edges_b = triangles[:, 2] - triangles[:, 0]
    face_cross = np.cross(face_edges_a, face_edges_b)
    face_normals = face_cross / np.maximum(
        np.linalg.norm(face_cross, axis=1)[:, None], 1e-30
    )
    output = 0
    for edge, segment_count in enumerate(segment_counts):
        face = int(face_indices[edge])
        start_local = int(edge_starts[edge])
        end_local = int(edge_ends[edge])
        count = int(segment_count)
        t = (
            np.arange(count, dtype=np.float32) + np.float32(0.5)
        ) / np.float32(count)
        end = output + count
        positions[output:end] = (
            start_positions[edge][None, :] * (1.0 - t[:, None])
            + end_positions[edge][None, :] * t[:, None]
        )
        normals[output:end] = face_normals[face]
        barycentrics[output:end, start_local] = 1.0 - t
        barycentrics[output:end, end_local] = t
        output_triangles[output:end] = triangle_indices[face]
        output = end
    return _InstanceCandidates(
        positions=positions,
        normals=normals,
        barycentrics=barycentrics,
        triangle_indices=output_triangles,
        surface_area=float(surface_area),
    )


def _deduplicate_audit_candidates(
    candidates: _InstanceCandidates,
    cell_size: float,
    normal_cosine_threshold: float,
) -> _InstanceCandidates:
    cells: dict[tuple[int, int, int], list[int]] = {}
    selected: list[int] = []
    inverse_cell_size = 1.0 / max(float(cell_size), 1e-8)
    coordinates = np.floor(
        candidates.positions.astype(np.float64) * inverse_cell_size
    ).astype(np.int64)
    for index, coordinate in enumerate(coordinates):
        key = (int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))
        representatives = cells.setdefault(key, [])
        normal = candidates.normals[index]
        if any(
            float(np.dot(normal, candidates.normals[representative]))
            >= float(normal_cosine_threshold)
            for representative in representatives
        ):
            continue
        representatives.append(index)
        selected.append(index)
    return _subset_candidates(
        candidates, np.asarray(selected, dtype=np.int64)
    )


def _exclude_selected_duplicates(
    candidates: _InstanceCandidates,
    base_candidates: _InstanceCandidates,
    base_selected: npt.NDArray[np.int64],
) -> _InstanceCandidates:
    occupied = {
        (
            base_candidates.positions[int(index)].tobytes(),
            base_candidates.normals[int(index)].tobytes(),
        )
        for index in base_selected
    }
    keep = []
    for index in range(candidates.positions.shape[0]):
        key = (
            candidates.positions[index].tobytes(),
            candidates.normals[index].tobytes(),
        )
        if key in occupied:
            continue
        occupied.add(key)
        keep.append(index)
    return _subset_candidates(candidates, np.asarray(keep, dtype=np.int64))


def _compatible_neighbor_contributions(
    index: int,
    positions: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    grid: _PointGrid,
    d_max: float,
    normal_cosine_threshold: float,
    plane_distance_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors = grid.nearby_indices(positions[index])
    if not neighbors:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )
    candidate_indices = np.asarray(neighbors, dtype=np.int64)
    candidate_indices = candidate_indices[candidate_indices != index]
    if candidate_indices.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )

    delta = positions[candidate_indices].astype(np.float64) - positions[
        index
    ].astype(np.float64)
    distance_squared = np.einsum("ij,ij->i", delta, delta)
    normal_dot = np.einsum(
        "ij,j->i",
        normals[candidate_indices].astype(np.float64),
        normals[index].astype(np.float64),
    )
    plane_limit = max(float(d_max) * float(plane_distance_scale), 1e-8)
    plane_a = np.abs(delta @ normals[index].astype(np.float64))
    plane_b = np.abs(
        np.einsum(
            "ij,ij->i",
            delta,
            normals[candidate_indices].astype(np.float64),
        )
    )
    valid = (
        (distance_squared > 1e-20)
        & (distance_squared < float(d_max) * float(d_max))
        & (normal_dot >= float(normal_cosine_threshold))
        & (plane_a <= plane_limit)
        & (plane_b <= plane_limit)
    )
    candidate_indices = candidate_indices[valid]
    distance_squared = distance_squared[valid]
    if candidate_indices.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )
    normalized_distance = np.sqrt(distance_squared) / float(d_max)
    contribution = np.power(
        np.maximum(1.0 - normalized_distance, 0.0), 8.0
    )
    return candidate_indices, contribution


def _weighted_sample_elimination_python(
    positions: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    output_count: int,
    *,
    surface_area: float,
    normal_cosine_threshold: float = 0.5,
    plane_distance_scale: float = 0.35,
    max_exact_candidates: int = 250_000,
) -> tuple[np.ndarray, float]:
    """Select a deterministic blue-noise-like subset using Yuksel-style WSE.

    This Python implementation is intentionally limited to prototype-sized
    inputs. The production 2.5M-candidate path needs the planned native backend.
    """

    input_count = int(positions.shape[0])
    output_count = max(1, min(int(output_count), input_count))
    if input_count == output_count:
        spacing = math.sqrt(max(float(surface_area), 1e-20) / output_count)
        return np.arange(input_count, dtype=np.int64), spacing
    if input_count > int(max_exact_candidates):
        raise ValueError(
            f"Prototype Weighted Sample Elimination supports at most "
            f"{max_exact_candidates:,} candidates, got {input_count:,}; "
            "use a smaller Cornell test budget until the native backend lands"
        )

    max_poisson_radius = math.sqrt(
        max(float(surface_area), 1e-20)
        / (2.0 * math.sqrt(3.0) * float(output_count))
    )
    d_max = max(2.0 * max_poisson_radius, 1e-6)
    active = np.ones((input_count,), dtype=np.bool_)
    grid = _PointGrid(positions, d_max, active)
    weights = np.zeros((input_count,), dtype=np.float64)
    for index in range(input_count):
        _, contribution = _compatible_neighbor_contributions(
            index,
            positions,
            normals,
            grid,
            d_max,
            normal_cosine_threshold,
            plane_distance_scale,
        )
        weights[index] = float(np.sum(contribution, dtype=np.float64))

    versions = np.zeros((input_count,), dtype=np.uint32)
    heap = [(-float(weights[i]), i, 0) for i in range(input_count)]
    heapq.heapify(heap)
    remaining = input_count
    while remaining > output_count:
        while heap:
            _, index, version = heapq.heappop(heap)
            if active[index] and int(versions[index]) == version:
                break
        else:
            raise RuntimeError("Weighted Sample Elimination heap exhausted")

        neighbors, contribution = _compatible_neighbor_contributions(
            index,
            positions,
            normals,
            grid,
            d_max,
            normal_cosine_threshold,
            plane_distance_scale,
        )
        active[index] = False
        remaining -= 1
        for neighbor, removed_weight in zip(neighbors, contribution):
            neighbor_index = int(neighbor)
            if not active[neighbor_index]:
                continue
            weights[neighbor_index] = max(
                0.0, float(weights[neighbor_index] - removed_weight)
            )
            versions[neighbor_index] += 1
            heapq.heappush(
                heap,
                (
                    -float(weights[neighbor_index]),
                    neighbor_index,
                    int(versions[neighbor_index]),
                ),
            )

    return np.flatnonzero(active).astype(np.int64), d_max * 0.5


def _adaptive_weighted_sample_elimination_python(
    positions: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    relative_densities: npt.NDArray[np.float32],
    output_count: int,
    *,
    surface_area: float,
    normal_cosine_threshold: float = 0.5,
    plane_distance_scale: float = 0.35,
    max_exact_candidates: int = 250_000,
) -> tuple[np.ndarray, float]:
    input_count = int(positions.shape[0])
    output_count = max(1, min(int(output_count), input_count))
    densities = np.asarray(relative_densities, dtype=np.float64)
    if densities.shape != (input_count,) or not np.all(
        np.isfinite(densities) & (densities > 0.0)
    ):
        raise ValueError(
            "relative_densities must be finite, positive, and shape (N,)"
        )
    base_poisson_radius = math.sqrt(
        max(float(surface_area), 1e-20)
        / (2.0 * math.sqrt(3.0) * float(output_count))
    )
    if input_count == output_count:
        return np.arange(input_count, dtype=np.int64), base_poisson_radius
    if input_count > int(max_exact_candidates):
        raise ValueError(
            f"Python adaptive WSE supports at most {max_exact_candidates:,} "
            f"candidates, got {input_count:,}"
        )

    base_d_max = max(2.0 * base_poisson_radius, 1e-6)
    local_d_max = np.maximum(base_d_max / np.sqrt(densities), 1e-6)
    active = np.ones((input_count,), dtype=np.bool_)
    grid = _PointGrid(positions, float(np.max(local_d_max)), active)
    weights = np.zeros((input_count,), dtype=np.float64)
    for index in range(input_count):
        _, contribution = _compatible_neighbor_contributions(
            index,
            positions,
            normals,
            grid,
            float(local_d_max[index]),
            normal_cosine_threshold,
            plane_distance_scale,
        )
        weights[index] = float(np.sum(contribution, dtype=np.float64))

    versions = np.zeros((input_count,), dtype=np.uint32)
    heap = [(-float(weights[i]), i, 0) for i in range(input_count)]
    heapq.heapify(heap)
    remaining = input_count
    while remaining > output_count:
        while heap:
            _, removed, version = heapq.heappop(heap)
            if active[removed] and int(versions[removed]) == version:
                break
        else:
            raise RuntimeError("Adaptive WSE heap exhausted")

        active[removed] = False
        remaining -= 1
        removed_position = positions[removed].astype(np.float64)
        removed_normal = normals[removed].astype(np.float64)
        for neighbor in grid.nearby_indices(positions[removed]):
            if not active[neighbor]:
                continue
            delta = removed_position - positions[neighbor].astype(np.float64)
            distance_squared = float(np.dot(delta, delta))
            d_max = float(local_d_max[neighbor])
            if distance_squared <= 1e-20 or distance_squared >= d_max * d_max:
                continue
            neighbor_normal = normals[neighbor].astype(np.float64)
            if float(np.dot(neighbor_normal, removed_normal)) < float(
                normal_cosine_threshold
            ):
                continue
            plane_limit = max(d_max * float(plane_distance_scale), 1e-8)
            if (
                abs(float(np.dot(delta, neighbor_normal))) > plane_limit
                or abs(float(np.dot(delta, removed_normal))) > plane_limit
            ):
                continue
            contribution = max(
                1.0 - math.sqrt(distance_squared) / d_max, 0.0
            ) ** 8
            weights[neighbor] = max(
                0.0, float(weights[neighbor] - contribution)
            )
            versions[neighbor] += 1
            heapq.heappush(
                heap,
                (
                    -float(weights[neighbor]),
                    neighbor,
                    int(versions[neighbor]),
                ),
            )

    return np.flatnonzero(active).astype(np.int64), base_poisson_radius


def weighted_sample_elimination(
    positions: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    output_count: int,
    *,
    surface_area: float,
    normal_cosine_threshold: float = 0.5,
    plane_distance_scale: float = 0.35,
    max_exact_candidates: int = 250_000,
    backend: SurfaceProbeSamplerBackend = "auto",
) -> tuple[np.ndarray, float]:
    resolved_backend = resolve_surface_probe_sampler_backend(backend)
    if resolved_backend == "cpp":
        from surface_probe_sampler import weighted_sample_elimination_cpp

        return weighted_sample_elimination_cpp(
            positions,
            normals,
            output_count,
            surface_area=surface_area,
            normal_cosine_threshold=normal_cosine_threshold,
            plane_distance_scale=plane_distance_scale,
        )

    return _weighted_sample_elimination_python(
        positions,
        normals,
        output_count,
        surface_area=surface_area,
        normal_cosine_threshold=normal_cosine_threshold,
        plane_distance_scale=plane_distance_scale,
        max_exact_candidates=max_exact_candidates,
    )


def adaptive_weighted_sample_elimination(
    positions: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    relative_densities: npt.NDArray[np.float32],
    output_count: int,
    *,
    surface_area: float,
    partition_masses: npt.ArrayLike | None = None,
    normal_cosine_threshold: float = 0.5,
    plane_distance_scale: float = 0.35,
    max_exact_candidates: int = 250_000,
    backend: SurfaceProbeSamplerBackend = "auto",
    profile_sink: list | None = None,
) -> tuple[np.ndarray, float]:
    resolved_backend = resolve_surface_probe_sampler_backend(backend)
    if resolved_backend == "cpp":
        from surface_probe_sampler import (
            adaptive_weighted_sample_elimination_cpp,
        )

        return adaptive_weighted_sample_elimination_cpp(
            positions,
            normals,
            relative_densities,
            output_count,
            surface_area=surface_area,
            partition_masses=partition_masses,
            normal_cosine_threshold=normal_cosine_threshold,
            plane_distance_scale=plane_distance_scale,
            profile_sink=profile_sink,
        )
    return _adaptive_weighted_sample_elimination_python(
        positions,
        normals,
        relative_densities,
        output_count,
        surface_area=surface_area,
        normal_cosine_threshold=normal_cosine_threshold,
        plane_distance_scale=plane_distance_scale,
        max_exact_candidates=max_exact_candidates,
    )


def resolve_surface_probe_sampler_backend(
    backend: SurfaceProbeSamplerBackend,
) -> Literal["cpp", "python"]:
    global _cpp_sampler_fallback_warning_printed

    if backend not in ("auto", "cpp", "python"):
        raise ValueError(f"Unknown surface probe sampler backend: {backend}")
    if backend == "python":
        return "python"
    try:
        from surface_probe_sampler import cpp_surface_probe_sampler_version

        cpp_surface_probe_sampler_version()
        return "cpp"
    except Exception as exc:
        from surface_probe_sampler import CppSurfaceProbeSamplerUnavailable

        if backend == "cpp" or not isinstance(
            exc, CppSurfaceProbeSamplerUnavailable
        ):
            raise
        if not _cpp_sampler_fallback_warning_printed:
            warnings.warn(
                "Native surface probe sampler is unavailable; falling back "
                f"to the Python reference implementation ({exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            _cpp_sampler_fallback_warning_printed = True
        return "python"


def deficit_repair(
    base_positions: npt.NDArray[np.float32],
    base_normals: npt.NDArray[np.float32],
    base_instances: npt.NDArray[np.uint32],
    candidate_positions: npt.NDArray[np.float32],
    candidate_normals: npt.NDArray[np.float32],
    candidate_instances: npt.NDArray[np.uint32],
    audit_positions: npt.NDArray[np.float32],
    audit_normals: npt.NDArray[np.float32],
    audit_instances: npt.NDArray[np.uint32],
    instance_radii: npt.NDArray[np.float32],
    *,
    min_gather_count: int,
    max_repair_count: int,
    normal_cosine_threshold: float,
    backend: SurfaceProbeSamplerBackend,
):
    resolved_backend = resolve_surface_probe_sampler_backend(backend)
    if resolved_backend == "cpp":
        from surface_probe_sampler import deficit_repair_cpp

        return deficit_repair_cpp(
            base_positions,
            base_normals,
            base_instances,
            candidate_positions,
            candidate_normals,
            candidate_instances,
            audit_positions,
            audit_normals,
            audit_instances,
            instance_radii,
            min_gather_count=min_gather_count,
            max_repair_count=max_repair_count,
            normal_cosine_threshold=normal_cosine_threshold,
        )
    from surface_probe_sampler import deficit_repair_python

    return deficit_repair_python(
        base_positions,
        base_normals,
        base_instances,
        candidate_positions,
        candidate_normals,
        candidate_instances,
        audit_positions,
        audit_normals,
        audit_instances,
        instance_radii,
        min_gather_count=min_gather_count,
        max_repair_count=max_repair_count,
        normal_cosine_threshold=normal_cosine_threshold,
    )


def estimate_surface_support(
    reference_positions: npt.NDArray[np.float32],
    reference_normals: npt.NDArray[np.float32],
    reference_instances: npt.NDArray[np.uint32],
    reference_area_weights: npt.NDArray[np.float32],
    query_positions: npt.NDArray[np.float32],
    query_normals: npt.NDArray[np.float32],
    query_instances: npt.NDArray[np.uint32],
    instance_radii: npt.NDArray[np.float32],
    *,
    normal_cosine_threshold: float,
    max_density_multiplier: float,
    backend: SurfaceProbeSamplerBackend,
):
    resolved_backend = resolve_surface_probe_sampler_backend(backend)
    if resolved_backend == "cpp":
        from surface_probe_sampler import estimate_support_cpp

        return estimate_support_cpp(
            reference_positions,
            reference_normals,
            reference_instances,
            reference_area_weights,
            query_positions,
            query_normals,
            query_instances,
            instance_radii,
            normal_cosine_threshold=normal_cosine_threshold,
            max_density_multiplier=max_density_multiplier,
        )
    from surface_probe_sampler import estimate_support_python

    return estimate_support_python(
        reference_positions,
        reference_normals,
        reference_instances,
        reference_area_weights,
        query_positions,
        query_normals,
        query_instances,
        instance_radii,
        normal_cosine_threshold=normal_cosine_threshold,
        max_density_multiplier=max_density_multiplier,
    )


def _make_probe_records(
    candidates: _InstanceCandidates,
    selected_indices: npt.NDArray[np.int64],
    radius: float,
    double_sided: bool,
    support_f: npt.NDArray[np.float32] | None = None,
    extra_flags: int = 0,
) -> np.ndarray:
    side_count = 2 if double_sided else 1
    records = np.zeros(
        (selected_indices.shape[0] * side_count,),
        dtype=SURFACE_PROBE_DTYPE,
    )
    positions = candidates.positions[selected_indices]
    normals = candidates.normals[selected_indices]
    barycentrics = candidates.barycentrics[selected_indices]
    triangles = candidates.triangle_indices[selected_indices]
    support_values = (
        np.ones((selected_indices.shape[0],), dtype=np.float32)
        if support_f is None
        else np.asarray(support_f, dtype=np.float32)
    )
    if support_values.shape != (selected_indices.shape[0],):
        raise ValueError("support_f must match the selected surface sites")
    for side_index in range(side_count):
        start = side_index * selected_indices.shape[0]
        end = start + selected_indices.shape[0]
        sign = -1.0 if side_index == 1 else 1.0
        records["position_radius"][start:end, :3] = positions
        records["position_radius"][start:end, 3] = float(radius)
        records["normal_side"][start:end, :3] = normals * sign
        records["normal_side"][start:end, 3] = support_values
        records["meta"][start:end, 0] = triangles
        records["meta"][start:end, 1] = barycentrics[:, 1].view(np.uint32)
        records["meta"][start:end, 2] = barycentrics[:, 2].view(np.uint32)
        records["meta"][start:end, 3] = np.uint32(
            int(extra_flags)
            | (SURFACE_PROBE_FLAG_BACK_SIDE if side_index == 1 else 0)
        )
    return records


def _build_point_octree_python(
    positions: npt.NDArray[np.float32],
    *,
    leaf_capacity: int,
    max_depth: int,
    profile_sink: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], float]:
    stage_start = time.perf_counter() if profile_sink is not None else 0.0

    def profile_mark(label: str) -> None:
        nonlocal stage_start
        if profile_sink is None:
            return
        now = time.perf_counter()
        profile_sink[label] = profile_sink.get(label, 0.0) + (
            now - stage_start
        )
        stage_start = now

    minimum = np.min(positions, axis=0).astype(np.float64)
    maximum = np.max(positions, axis=0).astype(np.float64)
    center = (minimum + maximum) * 0.5
    extent = max(float(np.max(maximum - minimum)) * 0.5, 1e-5)
    extent *= 1.0001
    profile_mark("bounds")
    nodes: list[list[int]] = [[0, 0, 0, 0]]
    probe_order: list[int] = []

    def fill_node(
        node_index: int,
        point_indices: np.ndarray,
        node_center: np.ndarray,
        node_extent: float,
        depth: int,
    ) -> None:
        if point_indices.size <= leaf_capacity or depth >= max_depth:
            probe_start = len(probe_order)
            probe_order.extend(int(index) for index in point_indices)
            nodes[node_index] = [0, 0, probe_start, int(point_indices.size)]
            return

        child_extent = node_extent * 0.5
        relative = positions[point_indices] > node_center.astype(np.float32)
        octants = (
            relative[:, 0].astype(np.uint8)
            | (relative[:, 1].astype(np.uint8) << 1)
            | (relative[:, 2].astype(np.uint8) << 2)
        )
        present_octants = [
            octant for octant in range(8) if np.any(octants == octant)
        ]
        child_base = len(nodes)
        nodes.extend([[0, 0, 0, 0] for _ in present_octants])
        child_mask = sum(1 << octant for octant in present_octants)
        nodes[node_index] = [child_base, child_mask, 0, 0]
        for compact_index, octant in enumerate(present_octants):
            offset = np.array(
                [
                    1.0 if octant & 1 else -1.0,
                    1.0 if octant & 2 else -1.0,
                    1.0 if octant & 4 else -1.0,
                ],
                dtype=np.float64,
            )
            fill_node(
                child_base + compact_index,
                point_indices[octants == octant],
                node_center + offset * child_extent,
                child_extent,
                depth + 1,
            )

    root_indices = np.arange(positions.shape[0], dtype=np.int64)
    profile_mark("root_indices")
    fill_node(
        0,
        root_indices,
        center,
        extent,
        0,
    )
    profile_mark("recursive_partition")
    node_array = np.asarray(nodes, dtype=np.uint32)
    probe_order_array = np.asarray(probe_order, dtype=np.int64)
    profile_mark("output_arrays")
    return (
        node_array,
        probe_order_array,
        (float(center[0]), float(center[1]), float(center[2])),
        float(extent),
    )


def _build_point_octree(
    positions: npt.NDArray[np.float32],
    *,
    leaf_capacity: int,
    max_depth: int,
    backend: SurfaceProbeSamplerBackend,
    profile_sink: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], float]:
    if backend == "cpp":
        from surface_probe_sampler import build_point_octree_cpp

        native_profiles = [] if profile_sink is not None else None
        result = build_point_octree_cpp(
            positions,
            leaf_capacity=leaf_capacity,
            max_depth=max_depth,
            profile_sink=native_profiles,
        )
        if profile_sink is not None and native_profiles:
            native = native_profiles[0]
            profile_sink["bounds"] = profile_sink.get("bounds", 0.0) + (
                native.bounds_ms / 1000.0
            )
            profile_sink["root_indices"] = profile_sink.get(
                "root_indices", 0.0
            ) + native.index_setup_ms / 1000.0
            profile_sink["recursive_partition"] = profile_sink.get(
                "recursive_partition", 0.0
            ) + native.partition_ms / 1000.0
            profile_sink["output_arrays"] = profile_sink.get(
                "output_arrays", 0.0
            ) + (native.flatten_ms + native.output_copy_ms) / 1000.0
            profile_sink["native_total"] = profile_sink.get(
                "native_total", 0.0
            ) + native.total_ms / 1000.0
            profile_sink["worker_count"] = max(
                profile_sink.get("worker_count", 0.0),
                float(native.worker_count),
            )
        return result
    return _build_point_octree_python(
        positions,
        leaf_capacity=leaf_capacity,
        max_depth=max_depth,
        profile_sink=profile_sink,
    )


@dataclass(frozen=True)
class SurfaceProbeLayout:
    probes: np.ndarray
    nodes: npt.NDArray[np.uint32]
    triangle_vertex_probes: npt.NDArray[np.uint32]
    instance_gpu_data: np.ndarray
    instance_infos: tuple[SurfaceProbeInstanceInfo, ...]
    target_surface_site_count: int
    total_base_surface_site_count: int
    total_repair_surface_site_count: int
    total_protected_surface_site_count: int
    repair_surface_site_budget: int
    total_surface_site_count: int
    total_candidate_count: int
    total_audit_point_count: int
    zero_gather_before: int
    zero_gather_after_repair: int
    zero_gather_after: int
    deficit_point_count_before: int
    deficit_point_count_after_repair: int
    deficit_point_count_after: int
    irreparable_audit_point_count: int
    repair_stop_reason: str
    protected_stop_reason: str
    ess_p50_before: float
    ess_p50_after: float
    support_f_p10: float
    support_f_p50: float
    density_m_p95: float
    max_density_multiplier: float
    adaptive_wse: bool
    total_probe_count: int
    total_vertex_anchor_site_count: int
    total_vertex_anchor_probe_count: int
    oversample_factor: int
    seed: int
    sampler_backend: Literal["cpp", "python"]
    build_profile: tuple[tuple[str, float], ...]

    @classmethod
    def build(
        cls,
        scene_node: SceneNode,
        *,
        target_probe_count: int = 8_192,
        oversample_factor: int = 5,
        seed: int = 1,
        leaf_capacity: int = 8,
        max_depth: int = 16,
        kernel_radius_scale: float = 2.5,
        normal_angle_degrees: float = 45.0,
        repair_budget_ratio: float = 0.30,
        repair_min_gather: int = 4,
        max_density_multiplier: float = 8.0,
        adaptive_wse: bool = True,
        build_vertex_anchors: bool = False,
        profile_build: bool = False,
        sampler_backend: SurfaceProbeSamplerBackend = "auto",
    ) -> "SurfaceProbeLayout":
        profile_origin = time.perf_counter()
        profile_previous = profile_origin
        profile_samples: list[tuple[str, float]] = []

        def profile_mark(label: str) -> None:
            nonlocal profile_previous
            now = time.perf_counter()
            elapsed = now - profile_previous
            profile_previous = now
            if profile_build:
                profile_samples.append((label, elapsed))
                print(
                    f"[SurfaceProbeProfile] {label}: {elapsed:.3f}s "
                    f"(total={now - profile_origin:.3f}s)",
                    flush=True,
                )

        target_probe_count = max(1, int(target_probe_count))
        oversample_factor = max(1, int(oversample_factor))
        leaf_capacity = max(1, int(leaf_capacity))
        max_depth = max(1, int(max_depth))
        kernel_radius_scale = max(float(kernel_radius_scale), 0.5)
        repair_budget_ratio = max(float(repair_budget_ratio), 0.0)
        repair_min_gather = max(1, min(32, int(repair_min_gather)))
        build_vertex_anchors = bool(build_vertex_anchors)
        max_density_multiplier = max(1.0, float(max_density_multiplier))
        normal_cosine_threshold = math.cos(
            math.radians(max(0.0, min(89.0, float(normal_angle_degrees))))
        )
        resolved_sampler_backend = resolve_surface_probe_sampler_backend(
            sampler_backend
        )

        triangle_sets = []
        areas = []
        valid_instance_indices = []
        for instance_index in range(len(scene_node.instances)):
            triangles, triangle_indices, mesh_triangle_indices, area = _instance_triangles(
                scene_node, instance_index
            )
            if area <= 0.0:
                continue
            valid_instance_indices.append(instance_index)
            triangle_sets.append(
                (triangles, triangle_indices, mesh_triangle_indices)
            )
            areas.append(area)
        if not valid_instance_indices:
            raise ValueError("Surface probes require at least one valid triangle")
        profile_mark("collect_geometry")

        area_array = np.asarray(areas, dtype=np.float64)
        global_spacing = math.sqrt(
            max(float(np.sum(area_array)), 1e-20) / target_probe_count
        )
        global_kernel_radius = max(
            global_spacing * kernel_radius_scale, 1e-5
        )
        preliminary_target_counts = _allocate_counts(
            area_array,
            target_probe_count,
            minimum_per_entry=min(
                repair_min_gather,
                max(1, target_probe_count // len(valid_instance_indices)),
            ),
        )
        adaptive_prepasses: list[_AdaptiveInstancePrepass] = []
        for valid_index, instance_index in enumerate(valid_instance_indices):
            triangles, triangle_indices, mesh_triangle_indices = triangle_sets[
                valid_index
            ]
            edge_a = triangles[:, 1] - triangles[:, 0]
            edge_b = triangles[:, 2] - triangles[:, 0]
            triangle_areas = (
                0.5
                * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
            ).astype(np.float64)
            centroids = _triangle_centroid_candidates(
                triangles, triangle_indices, float(area_array[valid_index])
            )
            if adaptive_wse:
                preliminary_count = int(
                    preliminary_target_counts[valid_index]
                )
                reference_count = max(
                    preliminary_count,
                    preliminary_count * min(oversample_factor, 2),
                )
                references = _sample_instance_surface(
                    triangles,
                    triangle_indices,
                    reference_count,
                    int(seed) + instance_index * 1_000_003 + 17,
                )
                support_references = _combine_candidate_sets(
                    [references, centroids], float(area_array[valid_index])
                )
                reference_area_weights = np.concatenate(
                    (
                        np.full(
                            (references.positions.shape[0],),
                            0.5
                            * float(area_array[valid_index])
                            / references.positions.shape[0],
                            dtype=np.float32,
                        ),
                        (0.5 * triangle_areas).astype(np.float32),
                    )
                )
                zero_instances = np.zeros(
                    (support_references.positions.shape[0],), dtype=np.uint32
                )
                query_instances = np.zeros(
                    (centroids.positions.shape[0],), dtype=np.uint32
                )
                triangle_support = estimate_surface_support(
                    support_references.positions,
                    support_references.normals,
                    zero_instances,
                    reference_area_weights,
                    centroids.positions,
                    centroids.normals,
                    query_instances,
                    np.asarray([global_kernel_radius], dtype=np.float32),
                    normal_cosine_threshold=normal_cosine_threshold,
                    max_density_multiplier=max_density_multiplier,
                    backend=resolved_sampler_backend,
                )
                triangle_densities = triangle_support.density_m
            else:
                triangle_densities = np.ones(
                    (triangles.shape[0],), dtype=np.float32
                )
            adaptive_mass = float(
                np.sum(
                    triangle_areas * triangle_densities.astype(np.float64),
                    dtype=np.float64,
                )
            )
            adaptive_prepasses.append(
                _AdaptiveInstancePrepass(
                    instance_index=instance_index,
                    triangles=triangles,
                    triangle_indices=triangle_indices,
                    mesh_triangle_indices=mesh_triangle_indices,
                    triangle_areas=triangle_areas,
                    surface_area=float(area_array[valid_index]),
                    triangle_densities=triangle_densities,
                    adaptive_mass=adaptive_mass,
                )
            )
        profile_mark("adaptive_support_prepass")

        target_counts = _allocate_counts(
            np.asarray(
                [state.adaptive_mass for state in adaptive_prepasses],
                dtype=np.float64,
            ),
            target_probe_count,
            minimum_per_entry=min(
                repair_min_gather,
                max(1, target_probe_count // len(valid_instance_indices)),
            ),
        )
        build_states: list[_InstanceBuildState] = []
        total_candidates = 0
        candidate_stage_origin = time.perf_counter()
        candidate_profile_seconds = {
            "surface_sampling": 0.0,
            "candidate_assembly": 0.0,
            "native_wse": 0.0,
            "boundary_candidates": 0.0,
            "inset_candidates": 0.0,
            "audit_combine": 0.0,
            "filter_partition": 0.0,
            "audit_deduplicate": 0.0,
            "repair_exclude": 0.0,
            "filter_compact": 0.0,
            "filter_subset_copy": 0.0,
            "support_and_metadata": 0.0,
        }
        wse_instance_samples: list[tuple[float, int, int, int]] = []
        native_wse_profiles: list[tuple[int, object]] = []
        candidate_filter_workers = 0
        candidate_filter_shards = 0

        valid_lookup = {
            instance_index: valid_index
            for valid_index, instance_index in enumerate(valid_instance_indices)
        }
        for instance_index in range(len(scene_node.instances)):
            if instance_index not in valid_lookup:
                raise ValueError(
                    f"Scene instance {instance_index} has no valid surface area"
                )
            valid_index = valid_lookup[instance_index]
            target_count = int(target_counts[valid_index])
            candidate_count = max(target_count, target_count * oversample_factor)
            prepass = adaptive_prepasses[valid_index]
            triangles = prepass.triangles
            triangle_indices = prepass.triangle_indices
            mesh_triangle_indices = prepass.mesh_triangle_indices
            substage_start = time.perf_counter()
            random_candidates = _sample_instance_surface(
                triangles,
                triangle_indices,
                candidate_count,
                int(seed) + instance_index * 1_000_003,
                prepass.triangle_densities if adaptive_wse else None,
            )
            random_local_triangles = np.searchsorted(
                triangle_indices, random_candidates.triangle_indices
            )
            random_densities = prepass.triangle_densities[
                random_local_triangles
            ]
            centroids = _triangle_centroid_candidates(
                triangles,
                triangle_indices,
                prepass.surface_area,
            )
            candidate_profile_seconds["surface_sampling"] += (
                time.perf_counter() - substage_start
            )
            substage_start = time.perf_counter()
            if adaptive_wse:
                base_candidates = _combine_candidate_sets(
                    [random_candidates, centroids], prepass.surface_area
                )
                candidate_densities = np.concatenate(
                    (random_densities, prepass.triangle_densities)
                ).astype(np.float32, copy=False)
            else:
                base_candidates = random_candidates
                candidate_densities = np.ones(
                    (base_candidates.positions.shape[0],), dtype=np.float32
                )
            adaptive_density_mean = max(
                prepass.adaptive_mass / prepass.surface_area, 1.0e-6
            )
            candidate_relative_densities = (
                candidate_densities / adaptive_density_mean
                if adaptive_wse
                else candidate_densities
            ).astype(np.float32, copy=False)
            candidate_profile_seconds["candidate_assembly"] += (
                time.perf_counter() - substage_start
            )
            substage_start = time.perf_counter()
            if adaptive_wse:
                native_profile_sink = [] if profile_build else None
                base_selected, poisson_radius = (
                    adaptive_weighted_sample_elimination(
                        base_candidates.positions,
                        base_candidates.normals,
                        candidate_relative_densities,
                        target_count,
                        surface_area=base_candidates.surface_area,
                        # Random candidates are already proposed with q(x) proportional
                        # to m(x). Equal partition mass avoids applying that density a
                        # second time while local adaptive radii still shape the WSE.
                        partition_masses=np.ones_like(
                            candidate_relative_densities
                        ),
                        normal_cosine_threshold=normal_cosine_threshold,
                        backend=resolved_sampler_backend,
                        profile_sink=native_profile_sink,
                    )
                )
                if native_profile_sink:
                    native_wse_profiles.append(
                        (instance_index, native_profile_sink[0])
                    )
            else:
                base_selected, poisson_radius = weighted_sample_elimination(
                    base_candidates.positions,
                    base_candidates.normals,
                    target_count,
                    surface_area=base_candidates.surface_area,
                    normal_cosine_threshold=normal_cosine_threshold,
                    backend=resolved_sampler_backend,
                )
            wse_elapsed = time.perf_counter() - substage_start
            candidate_profile_seconds["native_wse"] += wse_elapsed
            wse_instance_samples.append(
                (
                    wse_elapsed,
                    instance_index,
                    int(base_candidates.positions.shape[0]),
                    target_count,
                )
            )
            substage_start = time.perf_counter()
            spacing = math.sqrt(prepass.surface_area / target_count)
            kernel_radius = max(
                spacing * kernel_radius_scale,
                poisson_radius * kernel_radius_scale,
                global_kernel_radius,
                1e-5,
            )
            substage_start = time.perf_counter()
            boundary = _boundary_edge_candidates(
                triangles,
                triangle_indices,
                mesh_triangle_indices,
                kernel_radius * 0.5,
                base_candidates.surface_area,
            )
            candidate_profile_seconds["boundary_candidates"] += (
                time.perf_counter() - substage_start
            )
            substage_start = time.perf_counter()
            inset_corners = _triangle_inset_corner_candidates(
                triangles,
                triangle_indices,
                kernel_radius * 0.05,
                base_candidates.surface_area,
            )
            candidate_profile_seconds["inset_candidates"] += (
                time.perf_counter() - substage_start
            )
            substage_start = time.perf_counter()
            audit_source = _combine_candidate_sets(
                [boundary, inset_corners, base_candidates]
                if adaptive_wse
                else [boundary, inset_corners, centroids, base_candidates],
                base_candidates.surface_area,
            )
            candidate_profile_seconds["audit_combine"] += (
                time.perf_counter() - substage_start
            )
            if resolved_sampler_backend == "cpp":
                from surface_probe_sampler import (
                    filter_audit_repair_candidates_cpp,
                )

                filter_result = filter_audit_repair_candidates_cpp(
                    audit_source.positions,
                    audit_source.normals,
                    base_candidates.positions,
                    base_candidates.normals,
                    base_selected,
                    audit_cell_size=kernel_radius * 0.5,
                    normal_cosine_threshold=normal_cosine_threshold,
                )
                filter_profile = filter_result.profile
                candidate_profile_seconds["filter_partition"] += (
                    filter_profile.audit_partition_ms
                    + filter_profile.repair_partition_ms
                ) / 1000.0
                candidate_profile_seconds["audit_deduplicate"] += (
                    filter_profile.audit_deduplicate_ms / 1000.0
                )
                candidate_profile_seconds["repair_exclude"] += (
                    filter_profile.repair_exclude_ms / 1000.0
                )
                candidate_profile_seconds["filter_compact"] += (
                    filter_profile.compact_ms / 1000.0
                )
                candidate_filter_workers = max(
                    candidate_filter_workers, filter_profile.worker_count
                )
                candidate_filter_shards = max(
                    candidate_filter_shards, filter_profile.shard_count
                )
                substage_start = time.perf_counter()
                audit_candidates = _subset_candidates(
                    audit_source, filter_result.audit_indices
                )
                repair_candidates = _subset_candidates(
                    audit_source, filter_result.repair_indices
                )
                candidate_profile_seconds["filter_subset_copy"] += (
                    time.perf_counter() - substage_start
                )
            else:
                substage_start = time.perf_counter()
                audit_candidates = _deduplicate_audit_candidates(
                    audit_source,
                    kernel_radius * 0.5,
                    normal_cosine_threshold,
                )
                candidate_profile_seconds["audit_deduplicate"] += (
                    time.perf_counter() - substage_start
                )
                substage_start = time.perf_counter()
                repair_candidates = _exclude_selected_duplicates(
                    audit_source,
                    base_candidates,
                    base_selected,
                )
                candidate_profile_seconds["repair_exclude"] += (
                    time.perf_counter() - substage_start
                )
            substage_start = time.perf_counter()
            if adaptive_wse:
                support_candidates = base_candidates
                random_importance = 1.0 / np.maximum(
                    random_densities.astype(np.float64), 1.0e-12
                )
                random_importance *= (
                    0.5
                    * prepass.surface_area
                    / max(float(np.sum(random_importance)), 1.0e-30)
                )
                support_area_weights = np.concatenate(
                    (
                        random_importance.astype(np.float32),
                        (0.5 * prepass.triangle_areas).astype(np.float32),
                    )
                )
            else:
                support_candidates = _combine_candidate_sets(
                    [base_candidates, centroids],
                    base_candidates.surface_area,
                )
                support_area_weights = np.concatenate(
                    (
                        np.full(
                            (base_candidates.positions.shape[0],),
                            0.5
                            * base_candidates.surface_area
                            / base_candidates.positions.shape[0],
                            dtype=np.float32,
                        ),
                        (0.5 * prepass.triangle_areas).astype(np.float32),
                    )
                )
            _, material_id, _ = scene_node.instances[instance_index]
            if build_vertex_anchors:
                vertex_anchors, triangle_vertex_anchor_indices = (
                    _instance_vertex_anchors(
                        scene_node,
                        instance_index,
                        triangle_indices,
                        mesh_triangle_indices,
                        prepass.surface_area,
                    )
                )
            else:
                vertex_anchors = _combine_candidate_sets(
                    [], prepass.surface_area
                )
                triangle_vertex_anchor_indices = np.full(
                    (scene_node.meshes[
                        scene_node.instances[instance_index][0]
                    ].triangle_count, 3),
                    np.iinfo(np.uint32).max,
                    dtype=np.uint32,
                )
            build_states.append(
                _InstanceBuildState(
                    instance_index=instance_index,
                    base_candidates=base_candidates,
                    base_selected=base_selected,
                    audit_candidates=audit_candidates,
                    repair_candidates=repair_candidates,
                    support_candidates=support_candidates,
                    support_area_weights=support_area_weights,
                    candidate_relative_densities=(
                        candidate_relative_densities
                    ),
                    adaptive_density_mean=adaptive_density_mean,
                    adaptive_density_p95=float(
                        np.percentile(prepass.triangle_densities, 95.0)
                    ),
                    kernel_radius=kernel_radius,
                    poisson_radius=poisson_radius,
                    target_count=target_count,
                    double_sided=bool(
                        scene_node.materials[material_id].double_sided
                    ),
                    vertex_anchors=vertex_anchors,
                    triangle_vertex_anchor_indices=(
                        triangle_vertex_anchor_indices
                    ),
                )
            )
            total_candidates += base_candidates.positions.shape[0]
            candidate_profile_seconds["support_and_metadata"] += (
                time.perf_counter() - substage_start
            )
        candidate_stage_total = time.perf_counter() - candidate_stage_origin
        if profile_build:
            timed_total = sum(candidate_profile_seconds.values())
            slowest_wse = max(wse_instance_samples, default=(0.0, -1, 0, 0))
            top_wse = sorted(wse_instance_samples, reverse=True)[:5]
            top_wse_text = ", ".join(
                f"{instance}:{elapsed:.3f}s({candidate_count:,}->{target_count:,})"
                for elapsed, instance, candidate_count, target_count in top_wse
            )
            print(
                "[SurfaceProbeCandidateProfile] "
                f"sampling={candidate_profile_seconds['surface_sampling']:.3f}s; "
                f"assembly={candidate_profile_seconds['candidate_assembly']:.3f}s; "
                f"wse={candidate_profile_seconds['native_wse']:.3f}s; "
                f"boundary={candidate_profile_seconds['boundary_candidates']:.3f}s; "
                f"inset={candidate_profile_seconds['inset_candidates']:.3f}s; "
                f"audit_combine={candidate_profile_seconds['audit_combine']:.3f}s; "
                f"filter_partition="
                f"{candidate_profile_seconds['filter_partition']:.3f}s; "
                f"audit_dedupe="
                f"{candidate_profile_seconds['audit_deduplicate']:.3f}s; "
                f"repair_exclude={candidate_profile_seconds['repair_exclude']:.3f}s; "
                f"filter_compact="
                f"{candidate_profile_seconds['filter_compact']:.3f}s; "
                f"filter_copy="
                f"{candidate_profile_seconds['filter_subset_copy']:.3f}s; "
                f"filter_workers={candidate_filter_workers}; "
                f"filter_shards={candidate_filter_shards}; "
                f"support_metadata="
                f"{candidate_profile_seconds['support_and_metadata']:.3f}s; "
                f"other={max(candidate_stage_total - timed_total, 0.0):.3f}s; "
                f"slowest_wse=instance {slowest_wse[1]} "
                f"{slowest_wse[0]:.3f}s "
                f"({slowest_wse[2]:,}->{slowest_wse[3]:,}); "
                f"top_wse=[{top_wse_text}]; "
                f"total={candidate_stage_total:.3f}s",
                flush=True,
            )
            if native_wse_profiles:
                profile_values = [item[1] for item in native_wse_profiles]
                sum_field = lambda name: sum(
                    float(getattr(item, name)) for item in profile_values
                )
                slowest_native = max(
                    native_wse_profiles,
                    key=lambda item: float(getattr(item[1], "total_ms")),
                )
                slowest = slowest_native[1]
                print(
                    "[SurfaceProbeWSEProfile] "
                    f"samples={len(profile_values)}; "
                    f"parallel={sum(bool(getattr(item, 'parallel_path')) for item in profile_values)}; "
                    f"setup={sum_field('setup_ms') / 1000.0:.3f}s; "
                    f"stage1_partition="
                    f"{sum_field('stage1_partition_ms') / 1000.0:.3f}s; "
                    f"stage1_wall="
                    f"{sum_field('stage1_eliminate_wall_ms') / 1000.0:.3f}s; "
                    f"stage2_partition="
                    f"{sum_field('stage2_partition_ms') / 1000.0:.3f}s; "
                    f"stage2_wall="
                    f"{sum_field('stage2_eliminate_wall_ms') / 1000.0:.3f}s; "
                    f"final="
                    f"{(sum_field('final_pack_ms') + sum_field('final_grid_ms') + sum_field('final_weights_ms') + sum_field('final_heap_ms')) / 1000.0:.3f}s; "
                    f"native_total={sum_field('total_ms') / 1000.0:.3f}s; "
                    f"stage1_cpu(pack/grid/weights/heap)="
                    f"{sum_field('stage1_pack_cpu_ms') / 1000.0:.3f}/"
                    f"{sum_field('stage1_grid_cpu_ms') / 1000.0:.3f}/"
                    f"{sum_field('stage1_weights_cpu_ms') / 1000.0:.3f}/"
                    f"{sum_field('stage1_heap_cpu_ms') / 1000.0:.3f}s; "
                    f"stage2_cpu(pack/grid/weights/heap)="
                    f"{sum_field('stage2_pack_cpu_ms') / 1000.0:.3f}/"
                    f"{sum_field('stage2_grid_cpu_ms') / 1000.0:.3f}/"
                    f"{sum_field('stage2_weights_cpu_ms') / 1000.0:.3f}/"
                    f"{sum_field('stage2_heap_cpu_ms') / 1000.0:.3f}s; "
                    f"final(pack/grid/weights/heap)="
                    f"{sum_field('final_pack_ms') / 1000.0:.3f}/"
                    f"{sum_field('final_grid_ms') / 1000.0:.3f}/"
                    f"{sum_field('final_weights_ms') / 1000.0:.3f}/"
                    f"{sum_field('final_heap_ms') / 1000.0:.3f}s; "
                    f"slowest=instance {slowest_native[0]} "
                    f"{float(getattr(slowest, 'total_ms')) / 1000.0:.3f}s "
                    f"path={int(bool(getattr(slowest, 'parallel_path')))} "
                    f"counts={int(getattr(slowest, 'stage1_input_count')):,}->"
                    f"{int(getattr(slowest, 'stage1_output_count')):,}->"
                    f"{int(getattr(slowest, 'stage2_output_count')):,}->"
                    f"{int(getattr(slowest, 'final_output_count')):,}",
                    flush=True,
                )
        profile_mark("candidate_generation_and_wse")

        def concatenate_state_points(attribute: str, selected_base: bool):
            arrays = []
            instances = []
            offsets = [0]
            for state in build_states:
                candidates = getattr(state, attribute)
                values = (
                    candidates.positions[state.base_selected]
                    if selected_base
                    else candidates.positions
                )
                normals = (
                    candidates.normals[state.base_selected]
                    if selected_base
                    else candidates.normals
                )
                arrays.append((values, normals))
                instances.append(
                    np.full(
                        (values.shape[0],),
                        state.instance_index,
                        dtype=np.uint32,
                    )
                )
                offsets.append(offsets[-1] + values.shape[0])
            return (
                np.concatenate([item[0] for item in arrays]).astype(
                    np.float32, copy=False
                ),
                np.concatenate([item[1] for item in arrays]).astype(
                    np.float32, copy=False
                ),
                np.concatenate(instances),
                offsets,
            )

        base_positions, base_normals, base_instances, _ = (
            concatenate_state_points("base_candidates", True)
        )
        repair_positions, repair_normals, repair_instances, repair_offsets = (
            concatenate_state_points("repair_candidates", False)
        )
        audit_positions, audit_normals, audit_instances, audit_offsets = (
            concatenate_state_points("audit_candidates", False)
        )
        instance_radii = np.full(
            (len(scene_node.instances),), global_kernel_radius, dtype=np.float32
        )
        for state in build_states:
            instance_radii[state.instance_index] = state.kernel_radius
        profile_mark("audit_geometry")
        repair_budget = int(math.floor(target_probe_count * repair_budget_ratio))
        repair_result = deficit_repair(
            base_positions,
            base_normals,
            base_instances,
            repair_positions,
            repair_normals,
            repair_instances,
            audit_positions,
            audit_normals,
            audit_instances,
            instance_radii,
            min_gather_count=repair_min_gather,
            max_repair_count=repair_budget,
            normal_cosine_threshold=normal_cosine_threshold,
            backend=resolved_sampler_backend,
        )
        if profile_build and repair_result.profile is not None:
            repair_profile = repair_result.profile
            print(
                "[SurfaceProbeRepairProfile] "
                f"workers={repair_profile.worker_count}; "
                f"accel={repair_profile.acceleration_structure_ms / 1000.0:.3f}s; "
                f"base_gather={repair_profile.base_gather_ms / 1000.0:.3f}s; "
                f"coverage={repair_profile.coverage_build_ms / 1000.0:.3f}s "
                f"({repair_profile.coverage_pair_count:,} pairs); "
                f"heap={repair_profile.heap_build_ms / 1000.0:.3f}s; "
                f"greedy={repair_profile.greedy_select_ms / 1000.0:.3f}s; "
                f"affected={repair_profile.affected_audits_ms / 1000.0:.3f}s "
                f"({repair_profile.affected_audit_count:,} audits); "
                f"final_gather={repair_profile.final_gather_ms / 1000.0:.3f}s; "
                f"native_total={repair_profile.total_ms / 1000.0:.3f}s",
                flush=True,
            )
        repair_selected_global = np.zeros(
            (repair_positions.shape[0],), dtype=np.bool_
        )
        repair_selected_global[
            repair_result.selected_candidate_indices
        ] = True
        profile_mark("budgeted_deficit_repair")

        repair_selected_by_state: list[npt.NDArray[np.int64]] = []
        final_site_positions: list[npt.NDArray[np.float32]] = []
        final_site_normals: list[npt.NDArray[np.float32]] = []
        final_site_instances: list[npt.NDArray[np.uint32]] = []
        final_site_offsets = [0]
        for state_index, state in enumerate(build_states):
            repair_start = repair_offsets[state_index]
            repair_end = repair_offsets[state_index + 1]
            repair_selected = np.flatnonzero(
                repair_selected_global[repair_start:repair_end]
            ).astype(np.int64)
            repair_selected_by_state.append(repair_selected)
            positions = np.concatenate(
                (
                    state.base_candidates.positions[state.base_selected],
                    state.repair_candidates.positions[repair_selected],
                )
            )
            normals = np.concatenate(
                (
                    state.base_candidates.normals[state.base_selected],
                    state.repair_candidates.normals[repair_selected],
                )
            )
            final_site_positions.append(positions)
            final_site_normals.append(normals)
            final_site_instances.append(
                np.full(
                    (positions.shape[0],),
                    state.instance_index,
                    dtype=np.uint32,
                )
            )
            final_site_offsets.append(final_site_offsets[-1] + positions.shape[0])

        # Ordinary repair is budgeted. Coverage closure is not: every audit
        # point that is still below the gather target gets exactly its residual
        # deficit as colocated, face-local candidates.
        # These sites bypass WSE and are appended to the same reconstruction
        # octree as base and repair sites; they are not a second estimator.
        deficits_after_repair = np.maximum(
            repair_min_gather
            - repair_result.counts_after.astype(np.int64, copy=False),
            0,
        )
        deficit_after_repair = deficits_after_repair > 0
        protected_candidate_sets: list[_InstanceCandidates] = []
        protected_candidate_offsets = [0]
        for state_index, state in enumerate(build_states):
            audit_start = audit_offsets[state_index]
            audit_end = audit_offsets[state_index + 1]
            local_deficits = deficits_after_repair[audit_start:audit_end]
            local_deficit_indices = np.flatnonzero(
                local_deficits
            ).astype(np.int64)
            candidates = _repeat_candidate_subset_by_count(
                state.audit_candidates,
                local_deficit_indices,
                local_deficits[local_deficit_indices],
            )
            protected_candidate_sets.append(candidates)
            protected_candidate_offsets.append(
                protected_candidate_offsets[-1] + candidates.positions.shape[0]
            )

        protected_selected_by_state = [
            np.zeros((0,), dtype=np.int64) for _ in build_states
        ]
        counts_after_closure = repair_result.counts_after.copy()
        weight_sums_after_closure = repair_result.weight_sums_after.copy()
        ess_after_closure = repair_result.ess_after.copy()
        protected_candidate_count = protected_candidate_offsets[-1]
        if protected_candidate_count > 0:
            protected_positions = np.concatenate(
                [item.positions for item in protected_candidate_sets]
            ).astype(np.float32, copy=False)
            protected_normals = np.concatenate(
                [item.normals for item in protected_candidate_sets]
            ).astype(np.float32, copy=False)
            protected_instances = np.concatenate(
                [
                    np.full(
                        (item.positions.shape[0],),
                        state.instance_index,
                        dtype=np.uint32,
                    )
                    for state, item in zip(
                        build_states, protected_candidate_sets
                    )
                ]
            )
            closure_audit_indices = np.flatnonzero(
                deficit_after_repair
            ).astype(np.int64)
            closure_result = deficit_repair(
                np.concatenate(final_site_positions),
                np.concatenate(final_site_normals),
                np.concatenate(final_site_instances),
                protected_positions,
                protected_normals,
                protected_instances,
                audit_positions[closure_audit_indices],
                audit_normals[closure_audit_indices],
                audit_instances[closure_audit_indices],
                instance_radii,
                min_gather_count=repair_min_gather,
                max_repair_count=protected_candidate_count,
                normal_cosine_threshold=normal_cosine_threshold,
                backend=resolved_sampler_backend,
            )
            protected_selected_global = np.zeros(
                (protected_candidate_count,), dtype=np.bool_
            )
            protected_selected_global[
                closure_result.selected_candidate_indices
            ] = True
            for state_index, state in enumerate(build_states):
                candidate_start = protected_candidate_offsets[state_index]
                candidate_end = protected_candidate_offsets[state_index + 1]
                selected = np.flatnonzero(
                    protected_selected_global[candidate_start:candidate_end]
                ).astype(np.int64)
                protected_selected_by_state[state_index] = selected
                if selected.size == 0:
                    continue
                candidate_set = protected_candidate_sets[state_index]
                final_site_positions[state_index] = np.concatenate(
                    (
                        final_site_positions[state_index],
                        candidate_set.positions[selected],
                    )
                )
                final_site_normals[state_index] = np.concatenate(
                    (
                        final_site_normals[state_index],
                        candidate_set.normals[selected],
                    )
                )
                final_site_instances[state_index] = np.concatenate(
                    (
                        final_site_instances[state_index],
                        np.full(
                            (selected.size,),
                            state.instance_index,
                            dtype=np.uint32,
                        ),
                    )
                )
            counts_after_closure[closure_audit_indices] = (
                closure_result.counts_after
            )
            weight_sums_after_closure[closure_audit_indices] = (
                closure_result.weight_sums_after
            )
            ess_after_closure[closure_audit_indices] = closure_result.ess_after

        final_site_offsets = [0]
        for positions in final_site_positions:
            final_site_offsets.append(final_site_offsets[-1] + positions.shape[0])
        profile_mark("protected_deficit_closure")

        support_positions, support_normals, support_instances, _ = (
            concatenate_state_points("support_candidates", False)
        )
        support_area_weights = np.concatenate(
            [state.support_area_weights for state in build_states]
        ).astype(np.float32, copy=False)
        support_result = estimate_surface_support(
            support_positions,
            support_normals,
            support_instances,
            support_area_weights,
            np.concatenate(final_site_positions),
            np.concatenate(final_site_normals),
            np.concatenate(final_site_instances),
            instance_radii,
            normal_cosine_threshold=normal_cosine_threshold,
            max_density_multiplier=max_density_multiplier,
            backend=resolved_sampler_backend,
        )
        profile_mark("final_support_estimate")

        global_probes: list[np.ndarray] = []
        global_nodes: list[np.ndarray] = []
        global_triangle_vertex_probes: list[np.ndarray] = []
        instance_infos: list[SurfaceProbeInstanceInfo] = []
        instance_gpu_data = np.zeros(
            (len(scene_node.instances),), dtype=SURFACE_PROBE_INSTANCE_DTYPE
        )
        node_offset = 0
        probe_offset = 0
        triangle_anchor_offset = 0
        total_sites = 0
        total_protected_sites = 0
        total_vertex_anchor_sites = 0
        total_vertex_anchor_probes = 0
        pack_profile_seconds = {
            "probe_records": 0.0,
            "octree_build": 0.0,
            "octree_reorder_fixup": 0.0,
            "vertex_anchor_and_triangle_map": 0.0,
            "instance_metadata": 0.0,
            "global_concatenate": 0.0,
        }
        octree_instance_samples: list[tuple[float, int, int]] = []
        octree_profile_seconds: dict[str, float] = {}
        pack_substage_start = 0.0

        def pack_profile_mark(label: str) -> float:
            nonlocal pack_substage_start
            if not profile_build:
                return 0.0
            now = time.perf_counter()
            elapsed = now - pack_substage_start
            pack_profile_seconds[label] += elapsed
            pack_substage_start = now
            return elapsed

        for state_index, state in enumerate(build_states):
            if profile_build:
                pack_substage_start = time.perf_counter()
            instance_index = state.instance_index
            repair_selected = repair_selected_by_state[state_index]
            site_start = final_site_offsets[state_index]
            site_end = final_site_offsets[state_index + 1]
            instance_support_f = support_result.support_f[site_start:site_end]
            base_support_end = state.base_selected.shape[0]
            repair_support_end = base_support_end + repair_selected.shape[0]
            base_records = _make_probe_records(
                state.base_candidates,
                state.base_selected,
                state.kernel_radius,
                state.double_sided,
                instance_support_f[:base_support_end],
            )
            repair_records = _make_probe_records(
                state.repair_candidates,
                repair_selected,
                state.kernel_radius,
                state.double_sided,
                instance_support_f[base_support_end:repair_support_end],
            )
            protected_selected = protected_selected_by_state[state_index]
            protected_records = _make_probe_records(
                protected_candidate_sets[state_index],
                protected_selected,
                state.kernel_radius,
                state.double_sided,
                instance_support_f[repair_support_end:],
                extra_flags=SURFACE_PROBE_FLAG_PROTECTED,
            )
            reconstruction_records = np.concatenate(
                (base_records, repair_records, protected_records)
            )
            pack_profile_mark("probe_records")
            nodes, order, root_center, root_extent = _build_point_octree(
                reconstruction_records["position_radius"][:, :3],
                leaf_capacity=leaf_capacity,
                max_depth=max_depth,
                backend=resolved_sampler_backend,
                profile_sink=(
                    octree_profile_seconds if profile_build else None
                ),
            )
            octree_elapsed = pack_profile_mark("octree_build")
            if profile_build:
                octree_instance_samples.append(
                    (
                        octree_elapsed,
                        instance_index,
                        int(reconstruction_records.shape[0]),
                    )
                )
            reconstruction_records = reconstruction_records[order].copy()
            nodes = nodes.copy()
            branch_mask = nodes[:, 1] != 0
            nodes[branch_mask, 0] += np.uint32(node_offset)
            leaf_mask = ~branch_mask
            nodes[leaf_mask, 2] += np.uint32(probe_offset)
            pack_profile_mark("octree_reorder_fixup")

            anchor_site_count = int(state.vertex_anchors.positions.shape[0])
            anchor_records = _make_probe_records(
                state.vertex_anchors,
                np.arange(anchor_site_count, dtype=np.int64),
                state.kernel_radius,
                state.double_sided,
            )
            anchor_records["meta"][:, 3] |= np.uint32(
                SURFACE_PROBE_FLAG_VERTEX_ANCHOR
            )
            reconstruction_probe_count = int(
                reconstruction_records.shape[0]
            )
            anchor_probe_count = int(anchor_records.shape[0])
            records = np.concatenate(
                (reconstruction_records, anchor_records)
            )

            local_triangle_map = state.triangle_vertex_anchor_indices
            map_records = np.full(
                (local_triangle_map.shape[0] * 2, 4),
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
            # Bit 0 marks whether the record contains valid vertex anchors.
            map_records[:, 3] = 0
            valid_triangles = np.all(
                local_triangle_map != np.iinfo(np.uint32).max, axis=1
            )
            valid_triangle_indices = np.flatnonzero(valid_triangles)
            front_indices = (
                local_triangle_map[valid_triangles]
                + np.uint32(probe_offset + reconstruction_probe_count)
            )
            front_records = valid_triangle_indices * 2
            back_records = front_records + 1
            map_records[front_records, :3] = front_indices
            map_records[front_records, 3] |= np.uint32(1)
            back_indices = (
                front_indices + np.uint32(anchor_site_count)
                if state.double_sided
                else front_indices
            )
            map_records[back_records, :3] = back_indices
            map_records[back_records, 3] |= np.uint32(1)
            pack_profile_mark("vertex_anchor_and_triangle_map")

            node_count = int(nodes.shape[0])
            probe_count = int(records.shape[0])
            plane_sigma = max(state.kernel_radius * 0.25, 1e-6)
            audit_start = audit_offsets[state_index]
            audit_end = audit_offsets[state_index + 1]
            counts_before = repair_result.counts_before[audit_start:audit_end]
            counts_after_repair = repair_result.counts_after[
                audit_start:audit_end
            ]
            counts_after = counts_after_closure[audit_start:audit_end]
            ess_before = repair_result.ess_before[audit_start:audit_end]
            ess_after = ess_after_closure[audit_start:audit_end]
            repair_count = int(repair_selected.shape[0])
            protected_count = int(protected_selected.shape[0])
            final_site_count = (
                state.target_count + repair_count + protected_count
            )
            info = SurfaceProbeInstanceInfo(
                root_center=root_center,
                root_extent=root_extent,
                node_offset=node_offset,
                node_count=node_count,
                probe_offset=probe_offset,
                probe_count=probe_count,
                kernel_radius=state.kernel_radius,
                normal_cosine_threshold=normal_cosine_threshold,
                plane_sigma=plane_sigma,
                surface_area=state.base_candidates.surface_area,
                base_surface_site_count=state.target_count,
                repair_surface_site_count=repair_count,
                protected_surface_site_count=protected_count,
                surface_site_count=final_site_count,
                candidate_count=state.base_candidates.positions.shape[0],
                audit_point_count=state.audit_candidates.positions.shape[0],
                zero_gather_before=int(np.count_nonzero(counts_before == 0)),
                zero_gather_after_repair=int(
                    np.count_nonzero(counts_after_repair == 0)
                ),
                zero_gather_after=int(np.count_nonzero(counts_after == 0)),
                deficit_point_count_before=int(
                    np.count_nonzero(counts_before < repair_min_gather)
                ),
                deficit_point_count_after_repair=int(
                    np.count_nonzero(counts_after_repair < repair_min_gather)
                ),
                deficit_point_count_after=int(
                    np.count_nonzero(counts_after < repair_min_gather)
                ),
                ess_p50_before=float(np.median(ess_before)),
                ess_p50_after=float(np.median(ess_after)),
                support_f_p10=float(np.percentile(instance_support_f, 10.0)),
                support_f_p50=float(np.median(instance_support_f)),
                density_m_p95=float(
                    np.percentile(
                        support_result.density_m[site_start:site_end], 95.0
                    )
                ),
                adaptive_density_mean=state.adaptive_density_mean,
                adaptive_density_p95=state.adaptive_density_p95,
                reconstruction_probe_count=reconstruction_probe_count,
                vertex_anchor_site_count=anchor_site_count,
                vertex_anchor_probe_count=anchor_probe_count,
            )
            instance_infos.append(info)
            instance_gpu_data["root_center_extent"][instance_index] = (
                *root_center,
                root_extent,
            )
            instance_gpu_data["offsets"][instance_index] = (
                node_offset,
                node_count,
                probe_offset,
                probe_count,
            )
            instance_gpu_data["params"][instance_index] = (
                state.kernel_radius,
                normal_cosine_threshold,
                plane_sigma,
                np.asarray(triangle_anchor_offset, dtype=np.uint32)
                .view(np.float32)
                .item(),
            )
            global_probes.append(records)
            global_nodes.append(nodes)
            global_triangle_vertex_probes.append(map_records)
            node_offset += node_count
            probe_offset += probe_count
            triangle_anchor_offset += int(map_records.shape[0])
            total_sites += final_site_count
            total_protected_sites += protected_count
            total_vertex_anchor_sites += anchor_site_count
            total_vertex_anchor_probes += anchor_probe_count
            pack_profile_mark("instance_metadata")

        if profile_build:
            pack_substage_start = time.perf_counter()
        probes = np.concatenate(global_probes, axis=0)
        nodes = np.concatenate(global_nodes, axis=0)
        triangle_vertex_probes = np.concatenate(
            global_triangle_vertex_probes, axis=0
        )
        if probes.dtype.itemsize != SURFACE_PROBE_METADATA_SIZE:
            raise AssertionError("Surface probe GPU metadata stride mismatch")
        if nodes.shape[1] * nodes.dtype.itemsize != SURFACE_PROBE_NODE_SIZE:
            raise AssertionError("Surface probe GPU node stride mismatch")
        if instance_gpu_data.dtype.itemsize != SURFACE_PROBE_INSTANCE_SIZE:
            raise AssertionError("Surface probe GPU instance stride mismatch")
        pack_profile_mark("global_concatenate")
        if profile_build:
            slowest_octrees = sorted(
                octree_instance_samples, reverse=True
            )[:5]
            slowest_octree_text = ", ".join(
                f"{instance}:{elapsed:.3f}s({probe_count:,})"
                for elapsed, instance, probe_count in slowest_octrees
            )
            print(
                "[SurfaceProbePackProfile] "
                f"octree_backend={resolved_sampler_backend}; "
                f"octree_workers="
                f"{int(octree_profile_seconds.get('worker_count', 0.0))}; "
                f"records={pack_profile_seconds['probe_records']:.3f}s; "
                f"octrees={pack_profile_seconds['octree_build']:.3f}s; "
                f"octree_native="
                f"{octree_profile_seconds.get('native_total', 0.0):.3f}s; "
                f"octree_detail(bounds/root/recursive/output)="
                f"{octree_profile_seconds.get('bounds', 0.0):.3f}/"
                f"{octree_profile_seconds.get('root_indices', 0.0):.3f}/"
                f"{octree_profile_seconds.get('recursive_partition', 0.0):.3f}/"
                f"{octree_profile_seconds.get('output_arrays', 0.0):.3f}s; "
                f"reorder_fixup="
                f"{pack_profile_seconds['octree_reorder_fixup']:.3f}s; "
                f"anchors_maps="
                f"{pack_profile_seconds['vertex_anchor_and_triangle_map']:.3f}s; "
                f"instance_metadata="
                f"{pack_profile_seconds['instance_metadata']:.3f}s; "
                f"global_concat="
                f"{pack_profile_seconds['global_concatenate']:.3f}s; "
                f"slowest_octrees=[{slowest_octree_text}]",
                flush=True,
            )
        profile_mark("probe_pack_and_octrees")
        repair_residual_deficit_count = int(
            np.count_nonzero(repair_result.counts_after < repair_min_gather)
        )
        residual_deficit_count = int(
            np.count_nonzero(counts_after_closure < repair_min_gather)
        )
        selected_repair_count = int(
            repair_result.selected_candidate_indices.shape[0]
        )
        if repair_residual_deficit_count == 0:
            repair_stop_reason = "target_met"
        elif repair_budget <= 0:
            repair_stop_reason = "disabled"
        elif selected_repair_count >= repair_budget:
            repair_stop_reason = "budget_exhausted"
        else:
            repair_stop_reason = "candidates_exhausted"
        residual_zero_count = int(
            np.count_nonzero(counts_after_closure == 0)
        )
        if not np.any(deficit_after_repair):
            protected_stop_reason = "not_needed"
        elif residual_deficit_count == 0:
            protected_stop_reason = "target_met"
        else:
            protected_stop_reason = "irreparable"
        profile_mark("layout_finalize")
        if profile_build:
            layout_profile_total = sum(
                elapsed for _, elapsed in profile_samples
            )
            stage_text = ", ".join(
                f"{label}={elapsed:.3f}s/"
                f"{100.0 * elapsed / max(layout_profile_total, 1.0e-12):.1f}%"
                for label, elapsed in sorted(
                    profile_samples, key=lambda item: item[1], reverse=True
                )
            )
            print(
                "[SurfaceProbeLayoutProfile] "
                f"measured_total={layout_profile_total:.3f}s; "
                f"stages=[{stage_text}]",
                flush=True,
            )
        return cls(
            probes=probes,
            nodes=nodes,
            triangle_vertex_probes=triangle_vertex_probes,
            instance_gpu_data=instance_gpu_data,
            instance_infos=tuple(instance_infos),
            target_surface_site_count=target_probe_count,
            total_base_surface_site_count=target_probe_count,
            total_repair_surface_site_count=selected_repair_count,
            total_protected_surface_site_count=total_protected_sites,
            repair_surface_site_budget=repair_budget,
            total_surface_site_count=total_sites,
            total_candidate_count=total_candidates,
            total_audit_point_count=int(audit_positions.shape[0]),
            zero_gather_before=int(
                np.count_nonzero(repair_result.counts_before == 0)
            ),
            zero_gather_after_repair=int(
                np.count_nonzero(repair_result.counts_after == 0)
            ),
            zero_gather_after=residual_zero_count,
            deficit_point_count_before=int(
                np.count_nonzero(
                    repair_result.counts_before < repair_min_gather
                )
            ),
            deficit_point_count_after_repair=repair_residual_deficit_count,
            deficit_point_count_after=residual_deficit_count,
            irreparable_audit_point_count=residual_deficit_count,
            repair_stop_reason=repair_stop_reason,
            protected_stop_reason=protected_stop_reason,
            ess_p50_before=float(np.median(repair_result.ess_before)),
            ess_p50_after=float(np.median(ess_after_closure)),
            support_f_p10=float(np.percentile(support_result.support_f, 10.0)),
            support_f_p50=float(np.median(support_result.support_f)),
            density_m_p95=float(
                np.percentile(support_result.density_m, 95.0)
            ),
            max_density_multiplier=max_density_multiplier,
            adaptive_wse=bool(adaptive_wse),
            total_probe_count=int(probes.shape[0]),
            total_vertex_anchor_site_count=total_vertex_anchor_sites,
            total_vertex_anchor_probe_count=total_vertex_anchor_probes,
            oversample_factor=oversample_factor,
            seed=int(seed),
            sampler_backend=resolved_sampler_backend,
            build_profile=tuple(profile_samples),
        )

    def create_gpu_buffers(
        self,
        device: spy.Device,
        profile_sink: list[tuple[str, float]] | None = None,
    ) -> tuple[spy.Buffer, spy.Buffer, spy.Buffer, spy.Buffer]:
        stage_start = time.perf_counter() if profile_sink is not None else 0.0

        def profile_mark(label: str) -> None:
            nonlocal stage_start
            if profile_sink is None:
                return
            now = time.perf_counter()
            profile_sink.append((label, now - stage_start))
            stage_start = now

        probe_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="surface_probe_metadata",
            data=np.ascontiguousarray(self.probes).view(np.uint8),
        )
        profile_mark("gpu_probe_metadata_buffer")
        node_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="surface_probe_nodes",
            data=np.ascontiguousarray(self.nodes),
        )
        profile_mark("gpu_octree_buffer")
        instance_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="surface_probe_instances",
            data=np.ascontiguousarray(self.instance_gpu_data).view(np.uint8),
        )
        profile_mark("gpu_instance_buffer")
        triangle_vertex_probe_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="surface_probe_triangle_vertex_map",
            data=np.ascontiguousarray(self.triangle_vertex_probes),
        )
        profile_mark("gpu_triangle_vertex_map_buffer")
        return (
            probe_buffer,
            node_buffer,
            instance_buffer,
            triangle_vertex_probe_buffer,
        )

    @staticmethod
    def _sphere_intersects_cube(
        position: npt.NDArray[np.float64],
        radius: float,
        center: npt.NDArray[np.float64],
        extent: float,
    ) -> bool:
        distance = np.maximum(np.abs(position - center) - extent, 0.0)
        return float(np.dot(distance, distance)) <= radius * radius

    def query_probe_indices(
        self,
        instance_index: int,
        position: npt.ArrayLike,
        radius: float | None = None,
        normal: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        info = self.instance_infos[int(instance_index)]
        query_position = np.asarray(position, dtype=np.float64)
        query_radius = (
            info.kernel_radius if radius is None else max(float(radius), 0.0)
        )
        query_normal = (
            None
            if normal is None
            else np.asarray(normal, dtype=np.float64)
        )
        stack = [
            (
                info.node_offset,
                np.asarray(info.root_center, dtype=np.float64),
                info.root_extent,
            )
        ]
        result: list[int] = []
        while stack:
            node_index, center, extent = stack.pop()
            if not self._sphere_intersects_cube(
                query_position, query_radius, center, extent
            ):
                continue
            child_base, child_mask, probe_start, probe_count = (
                int(value) for value in self.nodes[node_index]
            )
            if child_mask == 0:
                for probe_index in range(
                    probe_start, probe_start + probe_count
                ):
                    probe_position = self.probes["position_radius"][
                        probe_index, :3
                    ].astype(np.float64)
                    delta = probe_position - query_position
                    if float(np.dot(delta, delta)) > query_radius * query_radius:
                        continue
                    if query_normal is not None:
                        probe_normal = self.probes["normal_side"][
                            probe_index, :3
                        ].astype(np.float64)
                        if (
                            float(np.dot(query_normal, probe_normal))
                            < info.normal_cosine_threshold
                        ):
                            continue
                    result.append(probe_index)
                continue

            child_extent = extent * 0.5
            compact_index = 0
            for octant in range(8):
                if not (child_mask & (1 << octant)):
                    continue
                offset = np.array(
                    [
                        1.0 if octant & 1 else -1.0,
                        1.0 if octant & 2 else -1.0,
                        1.0 if octant & 4 else -1.0,
                    ]
                )
                child_center = center + offset * child_extent
                if self._sphere_intersects_cube(
                    query_position,
                    query_radius,
                    child_center,
                    child_extent,
                ):
                    stack.append(
                        (
                            child_base + compact_index,
                            child_center,
                            child_extent,
                        )
                    )
                compact_index += 1
        return np.asarray(result, dtype=np.int64)
