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
    min_samples_per_mesh: int = 0,
) -> SurfaceSamples:
    sample_count = max(0, int(sample_count))
    min_samples_per_mesh = max(0, int(min_samples_per_mesh))
    if sample_count == 0 and min_samples_per_mesh == 0:
        return SurfaceSamples(
            positions=np.zeros((0, 3), dtype=np.float32),
            barycentrics=np.zeros((0, 3), dtype=np.float32),
            mesh_indices=np.zeros((0,), dtype=np.int32),
            triangle_indices=np.zeros((0,), dtype=np.int32),
        )

    mesh_pools = []

    for mesh_index, mesh in enumerate(model.meshes):
        tri = mesh.positions[mesh.indices.astype(np.int64)]
        p0 = tri[:, 0, :]
        p1 = tri[:, 1, :]
        p2 = tri[:, 2, :]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        valid = area > float(min_triangle_area)
        if not np.any(valid):
            continue

        mesh_pools.append(
            (
                p0[valid].astype(np.float32),
                (p1[valid] - p0[valid]).astype(np.float32),
                (p2[valid] - p0[valid]).astype(np.float32),
                area[valid].astype(np.float64),
                np.full(np.count_nonzero(valid), mesh_index, dtype=np.int32),
                np.nonzero(valid)[0].astype(np.int32),
            )
        )

    if not mesh_pools:
        raise ValueError("Cannot sample model surface: no non-degenerate triangles.")

    rng = np.random.default_rng(int(seed))

    def draw_from_pool(pool, count: int) -> SurfaceSamples:
        count = max(0, int(count))
        if count == 0:
            return SurfaceSamples(
                positions=np.zeros((0, 3), dtype=np.float32),
                barycentrics=np.zeros((0, 3), dtype=np.float32),
                mesh_indices=np.zeros((0,), dtype=np.int32),
                triangle_indices=np.zeros((0,), dtype=np.int32),
            )

        p0_all, edge01_all, edge02_all, areas, mesh_indices_all, triangle_indices_all = pool

        cdf = np.cumsum(areas, dtype=np.float64)
        cdf /= cdf[-1]
        cdf[-1] = 1.0

        triangle_choices = np.searchsorted(cdf, rng.random(count), side="right")
        triangle_choices = np.minimum(triangle_choices, cdf.shape[0] - 1)

        uv = rng.random((count, 2), dtype=np.float32)
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

    if min_samples_per_mesh == 0:
        merged_pool = (
            np.concatenate([pool[0] for pool in mesh_pools], axis=0),
            np.concatenate([pool[1] for pool in mesh_pools], axis=0),
            np.concatenate([pool[2] for pool in mesh_pools], axis=0),
            np.concatenate([pool[3] for pool in mesh_pools], axis=0),
            np.concatenate([pool[4] for pool in mesh_pools], axis=0),
            np.concatenate([pool[5] for pool in mesh_pools], axis=0),
        )
        return draw_from_pool(merged_pool, sample_count)

    mesh_count = len(mesh_pools)
    effective_sample_count = max(sample_count, mesh_count * min_samples_per_mesh)
    counts = np.full((mesh_count,), min_samples_per_mesh, dtype=np.int64)
    remaining = int(effective_sample_count - int(np.sum(counts)))
    if remaining > 0:
        mesh_areas = np.array([np.sum(pool[3], dtype=np.float64) for pool in mesh_pools], dtype=np.float64)
        weights = mesh_areas / np.sum(mesh_areas)
        expected = weights * float(remaining)
        extra = np.floor(expected).astype(np.int64)
        remainder = remaining - int(np.sum(extra))
        if remainder > 0:
            order = np.argsort(-(expected - extra))
            extra[order[:remainder]] += 1
        counts += extra

    parts = [draw_from_pool(pool, int(counts[index])) for index, pool in enumerate(mesh_pools)]

    return SurfaceSamples(
        positions=np.concatenate([part.positions for part in parts], axis=0).astype(np.float32),
        barycentrics=np.concatenate([part.barycentrics for part in parts], axis=0).astype(np.float32),
        mesh_indices=np.concatenate([part.mesh_indices for part in parts], axis=0),
        triangle_indices=np.concatenate([part.triangle_indices for part in parts], axis=0),
    )
