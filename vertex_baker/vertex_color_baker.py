from __future__ import annotations

import numpy as np
import numpy.typing as npt

from model import Material, Model
from surface_sampler import SurfaceSamples
from vertex_baking_utils import bake_least_squares, build_native


def _sample_material_nearest(material: Material, uv: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if material.base_color_texture is None:
        return material.base_color[:3].astype(np.float32)

    texture = material.base_color_texture
    h, w, _ = texture.shape
    u = float(uv[0] % 1.0)
    v = float(uv[1] % 1.0)
    x = min(w - 1, max(0, int(u * (w - 1))))
    y = min(h - 1, max(0, int(v * (h - 1))))
    return (texture[y, x, :3] * material.base_color[:3]).astype(np.float32)


def _sample_material_nearest_batch(material: Material, uvs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    uvs = np.asarray(uvs, dtype=np.float32)
    if uvs.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if material.base_color_texture is None:
        return np.broadcast_to(material.base_color[:3], (uvs.shape[0], 3)).astype(np.float32, copy=True)

    texture = material.base_color_texture
    h, w, _ = texture.shape
    u = np.mod(uvs[:, 0].astype(np.float64), 1.0)
    v = np.mod(uvs[:, 1].astype(np.float64), 1.0)
    x = np.clip((u * float(w - 1)).astype(np.int64), 0, w - 1)
    y = np.clip((v * float(h - 1)).astype(np.int64), 0, h - 1)
    return (texture[y, x, :3] * material.base_color[:3]).astype(np.float32)


def sample_base_color_values(model: Model, samples: SurfaceSamples) -> npt.NDArray[np.float32]:
    values = np.zeros((samples.positions.shape[0], 3), dtype=np.float32)
    for mesh_index, mesh in enumerate(model.meshes):
        mask = samples.mesh_indices == mesh_index
        if not np.any(mask):
            continue
        mesh = model.meshes[mesh_index]
        material = model.materials[mesh.material_index]
        triangles = mesh.indices[samples.triangle_indices[mask]].astype(np.int64)
        triangle_uvs = mesh.uvs[triangles]
        bary = samples.barycentrics[mask]
        uvs = (
            bary[:, 0, None] * triangle_uvs[:, 0]
            + bary[:, 1, None] * triangle_uvs[:, 1]
            + bary[:, 2, None] * triangle_uvs[:, 2]
        )
        values[mask] = _sample_material_nearest_batch(material, uvs)
    return values


def _clamp_to_sample_bounds(
    baked_values: npt.NDArray[np.float32],
    mesh_sample_values: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if mesh_sample_values.shape[0] == 0:
        return baked_values

    lower = np.min(mesh_sample_values, axis=0)
    upper = np.max(mesh_sample_values, axis=0)
    return np.clip(baked_values, lower, upper).astype(np.float32, copy=False)


def _vertex_anchor_samples(
    model: Model,
    mesh_index: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    mesh = model.meshes[mesh_index]
    material = model.materials[mesh.material_index]
    vertex_count = mesh.positions.shape[0]

    anchor_triangles = np.full(vertex_count, -1, dtype=np.int64)
    anchor_barycentrics = np.zeros((vertex_count, 3), dtype=np.float32)
    for triangle_index, triangle in enumerate(mesh.indices):
        for corner_index, vertex_index in enumerate(triangle.astype(np.int64)):
            if 0 <= vertex_index < vertex_count and anchor_triangles[vertex_index] < 0:
                anchor_triangles[vertex_index] = triangle_index
                anchor_barycentrics[vertex_index, corner_index] = 1.0

    used = anchor_triangles >= 0
    if not np.any(used):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    anchor_values = _sample_material_nearest_batch(material, mesh.uvs[used])

    return (
        np.nonzero(used)[0].astype(np.int64),
        anchor_triangles[used].astype(np.uint32),
        anchor_barycentrics[used].astype(np.float32, copy=False),
        anchor_values,
    )


def bake_model_vertex_colors(
    model: Model,
    samples: SurfaceSamples,
    sample_values: npt.NDArray[np.float32],
    *,
    regularization_weight: float = 0.0,
    build_native_first: bool = False,
    preserve_sample_bounds: bool = True,
    vertex_anchor_weight: int = 8,
    vertex_anchor_max_error: float = 0.08,
) -> list[npt.NDArray[np.float32]]:
    if build_native_first:
        build_native(force=True)

    sample_values = np.ascontiguousarray(sample_values, dtype=np.float32)
    if sample_values.ndim == 1:
        sample_values = sample_values.reshape(-1, 1)
    if sample_values.shape[0] != samples.positions.shape[0]:
        raise ValueError("sample_values length must match sample count")

    baked: list[npt.NDArray[np.float32]] = []
    for mesh_index, mesh in enumerate(model.meshes):
        mask = samples.mesh_indices == mesh_index
        anchor_vertices, anchor_triangles, anchor_barycentrics, anchor_values = _vertex_anchor_samples(model, mesh_index)
        if not np.any(mask):
            fallback = np.zeros((mesh.positions.shape[0], sample_values.shape[1]), dtype=np.float32)
            if anchor_vertices.shape[0] > 0:
                fallback[anchor_vertices] = anchor_values[:, : sample_values.shape[1]]
            baked.append(fallback)
            continue

        mesh_sample_values = sample_values[mask]
        mesh_sample_triangles = samples.triangle_indices[mask].astype(np.uint32)
        mesh_sample_barycentrics = samples.barycentrics[mask]

        anchor_repeat = max(0, int(vertex_anchor_weight))
        if anchor_repeat > 0:
            if anchor_triangles.shape[0] > 0:
                mesh_sample_triangles = np.concatenate(
                    [mesh_sample_triangles, np.tile(anchor_triangles, anchor_repeat)]
                )
                mesh_sample_barycentrics = np.concatenate(
                    [mesh_sample_barycentrics, np.tile(anchor_barycentrics, (anchor_repeat, 1))]
                )
                mesh_sample_values = np.concatenate(
                    [mesh_sample_values, np.tile(anchor_values, (anchor_repeat, 1))]
                )

        baked_values = bake_least_squares(
            mesh.positions,
            mesh.indices,
            mesh_sample_triangles,
            mesh_sample_barycentrics,
            mesh_sample_values,
            regularization_weight=regularization_weight,
        )
        if preserve_sample_bounds:
            baked_values = _clamp_to_sample_bounds(baked_values, mesh_sample_values)
        max_anchor_error = max(0.0, float(vertex_anchor_max_error))
        if max_anchor_error > 0.0 and anchor_vertices.shape[0] > 0:
            anchor_lower = anchor_values - max_anchor_error
            anchor_upper = anchor_values + max_anchor_error
            baked_values[anchor_vertices] = np.clip(baked_values[anchor_vertices], anchor_lower, anchor_upper)
        baked.append(baked_values.astype(np.float32, copy=False))
    return baked
