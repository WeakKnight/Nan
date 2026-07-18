from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


MESH_COLORS_EDGE_COUNT = 3
MESH_COLORS_INVALID_ADJACENCY = 0xFFFFFFFF
MESH_COLORS_ADJACENCY_FACE_SHIFT = 3
MESH_COLORS_ADJACENCY_EDGE_MASK = 0x3
MESH_COLORS_ADJACENCY_FLIP_BIT = 1 << 2
MESH_COLORS_ADJACENCY_MAX_FACE = (
    1 << (32 - MESH_COLORS_ADJACENCY_FACE_SHIFT)
) - 1


@dataclass(frozen=True)
class MeshColorsEdgeAdjacency:
    adjacent_face: int = MESH_COLORS_INVALID_ADJACENCY
    adjacent_edge: int = 0
    flip: bool = False

    @property
    def valid(self) -> bool:
        return self.adjacent_face != MESH_COLORS_INVALID_ADJACENCY

    def pack(self) -> int:
        if not self.valid:
            return MESH_COLORS_INVALID_ADJACENCY
        adjacent_face = int(self.adjacent_face)
        adjacent_edge = int(self.adjacent_edge)
        if adjacent_face < 0 or adjacent_face > MESH_COLORS_ADJACENCY_MAX_FACE:
            raise ValueError("adjacent face does not fit in the packed format")
        if adjacent_edge < 0 or adjacent_edge >= MESH_COLORS_EDGE_COUNT:
            raise ValueError("adjacent edge must be in [0, 2]")
        return (
            adjacent_face << MESH_COLORS_ADJACENCY_FACE_SHIFT
            | adjacent_edge
            | (MESH_COLORS_ADJACENCY_FLIP_BIT if self.flip else 0)
        )

    @classmethod
    def unpack(cls, packed: int) -> "MeshColorsEdgeAdjacency":
        packed = int(packed) & 0xFFFFFFFF
        if packed == MESH_COLORS_INVALID_ADJACENCY:
            return cls()
        adjacent_edge = packed & MESH_COLORS_ADJACENCY_EDGE_MASK
        if adjacent_edge >= MESH_COLORS_EDGE_COUNT:
            raise ValueError("packed adjacency contains an invalid edge")
        return cls(
            adjacent_face=packed >> MESH_COLORS_ADJACENCY_FACE_SHIFT,
            adjacent_edge=adjacent_edge,
            flip=bool(packed & MESH_COLORS_ADJACENCY_FLIP_BIT),
        )


@dataclass(frozen=True)
class MeshColorsFaceAdjacency:
    edges: tuple[
        MeshColorsEdgeAdjacency,
        MeshColorsEdgeAdjacency,
        MeshColorsEdgeAdjacency,
    ]

    def __getitem__(self, edge: int) -> MeshColorsEdgeAdjacency:
        return self.edges[int(edge)]

    def pack(self) -> bytes:
        return struct.pack(
            "IIII",
            self.edges[0].pack(),
            self.edges[1].pack(),
            self.edges[2].pack(),
            MESH_COLORS_INVALID_ADJACENCY,
        )


@dataclass(frozen=True)
class MeshColorsAdjacencyDiagnostics:
    boundary_edge_count: int = 0
    manifold_edge_count: int = 0
    non_manifold_edge_count: int = 0
    degenerate_face_count: int = 0
    orientation_anomaly_count: int = 0


@dataclass(frozen=True)
class MeshColorsMeshAdjacency:
    faces: tuple[MeshColorsFaceAdjacency, ...]
    diagnostics: MeshColorsAdjacencyDiagnostics

    def packed_uint4(self) -> npt.NDArray[np.uint32]:
        if not self.faces:
            return np.empty((0, 4), dtype=np.uint32)
        data = np.frombuffer(
            b"".join(face.pack() for face in self.faces),
            dtype=np.uint32,
        )
        return data.reshape(-1, 4).copy()


@dataclass(frozen=True)
class _EdgeUse:
    face: int
    edge: int
    start_vertex: int
    end_vertex: int


def triangle_edge_vertices(
    triangle: npt.ArrayLike,
    edge: int,
) -> tuple[int, int]:
    vertices = np.asarray(triangle).reshape(-1)
    if vertices.size != 3:
        raise ValueError("triangle must contain exactly three vertex indices")
    edge = int(edge)
    if edge < 0 or edge >= MESH_COLORS_EDGE_COUNT:
        raise ValueError("edge must be in [0, 2]")
    return int(vertices[edge]), int(vertices[(edge + 1) % 3])


def edge_barycentrics(edge: int, t: float) -> tuple[float, float, float]:
    """Return barycentrics on an edge, parameterized in vertex order."""
    edge = int(edge)
    t = float(t)
    if edge == 0:
        return 1.0 - t, t, 0.0
    if edge == 1:
        return 0.0, 1.0 - t, t
    if edge == 2:
        return t, 0.0, 1.0 - t
    raise ValueError("edge must be in [0, 2]")


def edge_lattice_ij(
    edge: int,
    t: float,
    resolution: int,
) -> tuple[float, float]:
    barycentrics = edge_barycentrics(edge, t)
    resolution = int(resolution)
    if resolution < 1:
        raise ValueError("resolution must be positive")
    return (
        barycentrics[0] * resolution,
        barycentrics[1] * resolution,
    )


def remap_edge_parameter(
    t: float,
    adjacency: MeshColorsEdgeAdjacency,
) -> float:
    if not adjacency.valid:
        raise ValueError("cannot remap across a boundary edge")
    t = float(t)
    return 1.0 - t if adjacency.flip else t


def build_triangle_adjacency(
    indices: npt.ArrayLike,
) -> MeshColorsMeshAdjacency:
    triangles = np.asarray(indices)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("indices must have shape (triangle_count, 3)")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("indices must use an integer dtype")

    face_count = int(triangles.shape[0])
    entries = [
        [MeshColorsEdgeAdjacency() for _ in range(MESH_COLORS_EDGE_COUNT)]
        for _ in range(face_count)
    ]
    edge_uses: dict[tuple[int, int], list[_EdgeUse]] = {}
    degenerate_face_count = 0

    for face, triangle in enumerate(triangles):
        vertex_ids = tuple(int(value) for value in triangle)
        if len(set(vertex_ids)) != 3:
            degenerate_face_count += 1
            continue
        for edge in range(MESH_COLORS_EDGE_COUNT):
            start_vertex, end_vertex = triangle_edge_vertices(triangle, edge)
            key = (
                min(start_vertex, end_vertex),
                max(start_vertex, end_vertex),
            )
            edge_uses.setdefault(key, []).append(
                _EdgeUse(
                    face=face,
                    edge=edge,
                    start_vertex=start_vertex,
                    end_vertex=end_vertex,
                )
            )

    boundary_edge_count = 0
    manifold_edge_count = 0
    non_manifold_edge_count = 0
    orientation_anomaly_count = 0
    for uses in edge_uses.values():
        if len(uses) == 1:
            boundary_edge_count += 1
            continue
        if len(uses) != 2 or uses[0].face == uses[1].face:
            non_manifold_edge_count += 1
            continue

        first, second = uses
        flip = (
            first.start_vertex == second.end_vertex
            and first.end_vertex == second.start_vertex
        )
        if not flip:
            orientation_anomaly_count += 1
        entries[first.face][first.edge] = MeshColorsEdgeAdjacency(
            adjacent_face=second.face,
            adjacent_edge=second.edge,
            flip=flip,
        )
        entries[second.face][second.edge] = MeshColorsEdgeAdjacency(
            adjacent_face=first.face,
            adjacent_edge=first.edge,
            flip=flip,
        )
        manifold_edge_count += 1

    return MeshColorsMeshAdjacency(
        faces=tuple(
            MeshColorsFaceAdjacency(
                edges=(face[0], face[1], face[2])
            )
            for face in entries
        ),
        diagnostics=MeshColorsAdjacencyDiagnostics(
            boundary_edge_count=boundary_edge_count,
            manifold_edge_count=manifold_edge_count,
            non_manifold_edge_count=non_manifold_edge_count,
            degenerate_face_count=degenerate_face_count,
            orientation_anomaly_count=orientation_anomaly_count,
        ),
    )
