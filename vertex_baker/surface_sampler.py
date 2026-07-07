from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from model import Model


@dataclass(frozen=True)
class SurfaceSamples:
    positions: npt.NDArray[np.float32]
    barycentrics: npt.NDArray[np.float32]
    mesh_indices: npt.NDArray[np.int32]
    triangle_indices: npt.NDArray[np.int32]


def sample_model_surface(
    model: Model,
    sample_count: int,
    seed: int = 1,
    min_triangle_area: float = 1e-12,
) -> SurfaceSamples:
    sample_count = max(0, int(sample_count))
    if sample_count == 0:
        return SurfaceSamples(
            positions=np.zeros((0, 3), dtype=np.float32),
            barycentrics=np.zeros((0, 3), dtype=np.float32),
            mesh_indices=np.zeros((0,), dtype=np.int32),
            triangle_indices=np.zeros((0,), dtype=np.int32),
        )

    p0_parts = []
    edge01_parts = []
    edge02_parts = []
    area_parts = []
    mesh_index_parts = []
    triangle_index_parts = []

    for mesh_index, mesh in enumerate(model.meshes):
        tri = mesh.positions[mesh.indices.astype(np.int64)]
        p0 = tri[:, 0, :]
        p1 = tri[:, 1, :]
        p2 = tri[:, 2, :]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        valid = area > float(min_triangle_area)
        if not np.any(valid):
            continue

        p0_parts.append(p0[valid])
        edge01_parts.append(p1[valid] - p0[valid])
        edge02_parts.append(p2[valid] - p0[valid])
        area_parts.append(area[valid].astype(np.float64))
        mesh_index_parts.append(np.full(np.count_nonzero(valid), mesh_index, dtype=np.int32))
        triangle_index_parts.append(np.nonzero(valid)[0].astype(np.int32))

    if not area_parts:
        raise ValueError("Cannot sample model surface: no non-degenerate triangles.")

    p0_all = np.concatenate(p0_parts, axis=0).astype(np.float32)
    edge01_all = np.concatenate(edge01_parts, axis=0).astype(np.float32)
    edge02_all = np.concatenate(edge02_parts, axis=0).astype(np.float32)
    mesh_indices_all = np.concatenate(mesh_index_parts, axis=0)
    triangle_indices_all = np.concatenate(triangle_index_parts, axis=0)
    areas = np.concatenate(area_parts, axis=0)

    cdf = np.cumsum(areas, dtype=np.float64)
    cdf /= cdf[-1]
    cdf[-1] = 1.0

    rng = np.random.default_rng(int(seed))
    triangle_choices = np.searchsorted(cdf, rng.random(sample_count), side="right")
    triangle_choices = np.minimum(triangle_choices, cdf.shape[0] - 1)

    uv = rng.random((sample_count, 2), dtype=np.float32)
    su = np.sqrt(uv[:, 0], dtype=np.float32)
    b0 = 1.0 - su
    b1 = su * (1.0 - uv[:, 1])
    b2 = su * uv[:, 1]
    barycentrics = np.stack([b0, b1, b2], axis=1).astype(np.float32)
    positions = (
        p0_all[triangle_choices]
        + edge01_all[triangle_choices] * b1[:, None]
        + edge02_all[triangle_choices] * b2[:, None]
    )

    return SurfaceSamples(
        positions=positions.astype(np.float32),
        barycentrics=barycentrics,
        mesh_indices=mesh_indices_all[triangle_choices],
        triangle_indices=triangle_indices_all[triangle_choices],
    )
