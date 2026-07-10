from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from model import Mesh, Model
from surface_sampler import SurfaceSamples, sample_model_surface


HALF_PI = 0.5 * np.pi


@dataclass(frozen=True)
class VisibilitySampleCones:
    directions: npt.NDArray[np.float32]
    aperture_radians: npt.NDArray[np.float32]
    scale: npt.NDArray[np.float32]
    visible_fraction: npt.NDArray[np.float32]


@dataclass(frozen=True)
class VisibilityBakeResult:
    samples: SurfaceSamples
    sample_cones: VisibilitySampleCones | None
    vertex_cones: list[npt.NDArray[np.float32]]
    encoded_texcoord2: list[npt.NDArray[np.float32]]
    sample_sh: npt.NDArray[np.float32] | None = None
    proxy_vertex_sh: npt.NDArray[np.float32] | None = None
    sample_normals: npt.NDArray[np.float32] | None = None
    samples_per_triangle: int | None = None
    proxy_triangle_count: int | None = None
    trace_backend: str | None = None
    trace_statistics: dict[str, object] | None = None


def _normalize_rows(
    values: npt.NDArray[np.float32],
    fallback: npt.NDArray[np.float32] | None = None,
    eps: float = 1e-8,
) -> npt.NDArray[np.float32]:
    values = np.asarray(values, dtype=np.float32)
    lengths = np.linalg.norm(values, axis=1)
    result = values.copy()
    valid = lengths > float(eps)
    result[valid] /= lengths[valid, None]
    if np.any(~valid):
        if fallback is None:
            result[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            fallback = np.asarray(fallback, dtype=np.float32)
            if fallback.ndim == 1:
                result[~valid] = fallback
            else:
                result[~valid] = fallback[~valid]
            fallback_lengths = np.linalg.norm(result[~valid], axis=1)
            valid_fallback = fallback_lengths > float(eps)
            if np.any(valid_fallback):
                invalid_indices = np.nonzero(~valid)[0]
                result[invalid_indices[valid_fallback]] /= fallback_lengths[valid_fallback, None]
            if np.any(~valid_fallback):
                invalid_indices = np.nonzero(~valid)[0]
                result[invalid_indices[~valid_fallback]] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return result.astype(np.float32, copy=False)


def _fallback_tangent(normals: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    normals = _normalize_rows(normals)
    helper = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (normals.shape[0], 1))
    near_z = np.abs(normals[:, 2]) > 0.9
    helper[near_z] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangents = np.cross(helper, normals)
    return _normalize_rows(tangents, np.array([1.0, 0.0, 0.0], dtype=np.float32))


def compute_mesh_tangents(mesh: Mesh) -> npt.NDArray[np.float32]:
    positions = np.asarray(mesh.positions, dtype=np.float32)
    normals = _normalize_rows(np.asarray(mesh.normals, dtype=np.float32))
    uvs = np.asarray(mesh.uvs, dtype=np.float32)
    indices = np.asarray(mesh.indices, dtype=np.uint32)

    tan1 = np.zeros_like(positions, dtype=np.float32)
    tan2 = np.zeros_like(positions, dtype=np.float32)
    for tri in indices.astype(np.int64):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0, p1, p2 = positions[i0], positions[i1], positions[i2]
        uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
        edge1 = p1 - p0
        edge2 = p2 - p0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0
        denom = float(duv1[0] * duv2[1] - duv1[1] * duv2[0])
        if abs(denom) <= 1e-10:
            tri_normal = np.cross(edge1, edge2).astype(np.float32)
            tangent = _fallback_tangent(tri_normal.reshape(1, 3))[0]
            bitangent = _normalize_rows(np.cross(tri_normal, tangent).reshape(1, 3), normals[i0])[0]
        else:
            inv = 1.0 / denom
            tangent = (edge1 * duv2[1] - edge2 * duv1[1]) * inv
            bitangent = (edge2 * duv1[0] - edge1 * duv2[0]) * inv
        for vertex_index in (i0, i1, i2):
            tan1[vertex_index] += tangent
            tan2[vertex_index] += bitangent

    fallback = _fallback_tangent(normals)
    tangent_xyz = tan1 - normals * np.sum(normals * tan1, axis=1, keepdims=True)
    tangent_xyz = _normalize_rows(tangent_xyz, fallback)
    handedness = np.where(np.sum(np.cross(normals, tangent_xyz) * tan2, axis=1) < 0.0, -1.0, 1.0)
    return np.concatenate([tangent_xyz, handedness[:, None].astype(np.float32)], axis=1).astype(np.float32)


def tangent_space_to_world(
    local_dirs: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    tangents: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    local_dirs = np.asarray(local_dirs, dtype=np.float32)
    normals = _normalize_rows(normals)
    tangent_xyz = _normalize_rows(tangents[:, :3], _fallback_tangent(normals))
    bitangents = _normalize_rows(np.cross(normals, tangent_xyz) * tangents[:, 3:4], _fallback_tangent(normals))
    return _normalize_rows(
        local_dirs[:, 0:1] * tangent_xyz + local_dirs[:, 1:2] * bitangents + local_dirs[:, 2:3] * normals,
        normals,
    )


def encode_visibility_cone_texcoord2(
    directions: npt.NDArray[np.float32],
    aperture_radians: npt.NDArray[np.float32],
    scale: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    tangents: npt.NDArray[np.float32],
    *,
    clamp_cone: bool = True,
) -> npt.NDArray[np.float32]:
    directions = _normalize_rows(directions, normals)
    normals = _normalize_rows(normals)
    tangent_xyz = _normalize_rows(tangents[:, :3], _fallback_tangent(normals))
    bitangents = _normalize_rows(np.cross(normals, tangent_xyz) * tangents[:, 3:4], _fallback_tangent(normals))
    ortho_tangents = _normalize_rows(np.cross(bitangents, normals), tangent_xyz)

    bitangent_component = np.sum(directions * bitangents, axis=1)
    normal_component = np.sum(directions * normals, axis=1)
    ortho_component = np.sum(directions * ortho_tangents, axis=1)
    tangent_angle = np.arctan2(ortho_component, normal_component)

    encoded = np.zeros((directions.shape[0], 4), dtype=np.float32)
    encoded[:, 0] = np.clip(bitangent_component, -1.0, 1.0).astype(np.float32)
    encoded[:, 1] = tangent_angle.astype(np.float32)
    if clamp_cone:
        encoded[:, 2] = np.clip(aperture_radians, 0.0, HALF_PI).astype(np.float32)
        encoded[:, 3] = np.clip(scale, 0.0, 1.0).astype(np.float32)
    else:
        encoded[:, 2] = np.asarray(aperture_radians, dtype=np.float32)
        encoded[:, 3] = np.asarray(scale, dtype=np.float32)
    return encoded


def decode_visibility_cone_texcoord2(
    encoded: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    tangents: npt.NDArray[np.float32],
) -> VisibilitySampleCones:
    encoded = np.asarray(encoded, dtype=np.float32)
    normals = _normalize_rows(normals)
    tangent_xyz = _normalize_rows(tangents[:, :3], _fallback_tangent(normals))
    bitangents = _normalize_rows(np.cross(normals, tangent_xyz) * tangents[:, 3:4], _fallback_tangent(normals))
    ortho_tangents = _normalize_rows(np.cross(bitangents, normals), tangent_xyz)

    bitangent_component = np.clip(encoded[:, 0], -1.0, 1.0)
    tangent_length = np.sqrt(np.maximum(1.0 - bitangent_component * bitangent_component, 0.0))
    tangent_angle = encoded[:, 1]
    directions = (
        bitangent_component[:, None] * bitangents
        + (
            np.cos(tangent_angle)[:, None] * normals
            + np.sin(tangent_angle)[:, None] * ortho_tangents
        )
        * tangent_length[:, None]
    )
    aperture_radians = np.clip(encoded[:, 2], 0.0, HALF_PI).astype(np.float32)
    scale = np.clip(encoded[:, 3], 0.0, 1.0).astype(np.float32)
    aperture_normalized = np.clip(aperture_radians / HALF_PI, 0.0, 1.0)
    visible_fraction = np.clip(aperture_normalized * scale, 0.0, 1.0).astype(np.float32)
    return VisibilitySampleCones(
        directions=_normalize_rows(directions, normals),
        aperture_radians=aperture_radians,
        scale=scale,
        visible_fraction=visible_fraction,
    )


def bake_least_squares_python(
    positions,
    indices,
    sample_triangles,
    sample_barycentrics,
    sample_values,
    *,
    diagonal_epsilon: float = 1e-10,
) -> npt.NDArray[np.float32]:
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.uint32)
    sample_triangles = np.ascontiguousarray(sample_triangles, dtype=np.uint32)
    sample_barycentrics = np.ascontiguousarray(sample_barycentrics, dtype=np.float32)
    sample_values = np.ascontiguousarray(sample_values, dtype=np.float32)
    if sample_values.ndim == 1:
        sample_values = sample_values.reshape(-1, 1)
    if sample_triangles.ndim != 1:
        raise ValueError("sample_triangles must have shape (N,)")
    if sample_barycentrics.shape != (sample_triangles.shape[0], 3):
        raise ValueError("sample_barycentrics must have shape (N, 3)")
    if sample_values.shape[0] != sample_triangles.shape[0]:
        raise ValueError("sample_values length must match sample count")

    vertex_count = int(positions.shape[0])
    channels = int(sample_values.shape[1])
    matrix = np.zeros((vertex_count, vertex_count), dtype=np.float64)
    rhs = np.zeros((vertex_count, channels), dtype=np.float64)
    lumped = np.zeros((vertex_count,), dtype=np.float64)

    for sample_index, triangle_index_raw in enumerate(sample_triangles):
        triangle_index = int(triangle_index_raw)
        if triangle_index < 0 or triangle_index >= indices.shape[0]:
            raise ValueError("sample triangle index out of range")
        tri_vertices = indices[triangle_index].astype(np.int64)
        bary = sample_barycentrics[sample_index].astype(np.float64)
        if abs(float(np.sum(bary)) - 1.0) > 1e-3:
            raise ValueError("sample barycentric coordinates must sum to one")
        if np.any(tri_vertices < 0) or np.any(tri_vertices >= vertex_count):
            raise ValueError("triangle vertex index out of range")
        if not np.all(np.isfinite(bary)):
            raise ValueError("non-finite barycentric coordinate")

        for i in range(3):
            vi = int(tri_vertices[i])
            bi = float(bary[i])
            lumped[vi] += bi
            for j in range(3):
                matrix[vi, int(tri_vertices[j])] += bi * float(bary[j])
            rhs[vi, :] += bi * sample_values[sample_index].astype(np.float64)

    for vertex_index in range(vertex_count):
        matrix[vertex_index, vertex_index] += 1.0 if lumped[vertex_index] <= 0.0 else float(diagonal_epsilon)

    return np.linalg.solve(matrix, rhs).astype(np.float32)


def _local_fibonacci_hemisphere(ray_count: int) -> npt.NDArray[np.float32]:
    ray_count = max(1, int(ray_count))
    golden_ratio = (1.0 + np.sqrt(5.0)) * 0.5
    i = np.arange(ray_count, dtype=np.float64)
    z = (i + 0.5) / float(ray_count)
    phi = 2.0 * np.pi * np.mod((i + 0.5) / golden_ratio, 1.0)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1).astype(np.float32)


def make_visibility_local_directions(ray_count: int) -> npt.NDArray[np.float32]:
    return _local_fibonacci_hemisphere(ray_count)


def _interpolate_mesh_vectors(
    mesh: Mesh,
    values: npt.NDArray[np.float32],
    sample_triangles: npt.NDArray[np.int32],
    barycentrics: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    tri_vertices = mesh.indices[sample_triangles.astype(np.int64)].astype(np.int64)
    tri_values = values[tri_vertices]
    return (
        barycentrics[:, 0:1] * tri_values[:, 0]
        + barycentrics[:, 1:2] * tri_values[:, 1]
        + barycentrics[:, 2:3] * tri_values[:, 2]
    ).astype(np.float32)


def _sample_frames(
    model: Model,
    samples: SurfaceSamples,
    mesh_tangents: list[npt.NDArray[np.float32]],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    normals = np.zeros_like(samples.positions, dtype=np.float32)
    tangents = np.zeros((samples.positions.shape[0], 4), dtype=np.float32)
    for mesh_index, mesh in enumerate(model.meshes):
        mask = samples.mesh_indices == mesh_index
        if not np.any(mask):
            continue
        normals[mask] = _interpolate_mesh_vectors(mesh, mesh.normals, samples.triangle_indices[mask], samples.barycentrics[mask])
        tangent_xyz = _interpolate_mesh_vectors(mesh, mesh_tangents[mesh_index][:, :3], samples.triangle_indices[mask], samples.barycentrics[mask])
        tangent_w = _interpolate_mesh_vectors(
            mesh,
            mesh_tangents[mesh_index][:, 3:4],
            samples.triangle_indices[mask],
            samples.barycentrics[mask],
        )[:, 0]
        tangents[mask, :3] = tangent_xyz
        tangents[mask, 3] = np.where(tangent_w < 0.0, -1.0, 1.0)
    normals = _normalize_rows(normals)
    tangents[:, :3] = _normalize_rows(tangents[:, :3], _fallback_tangent(normals))
    tangents[:, 3] = np.where(tangents[:, 3] < 0.0, -1.0, 1.0)
    return normals, tangents


def compute_visibility_sample_frames(
    model: Model,
    samples: SurfaceSamples,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    return _sample_frames(model, samples, [compute_mesh_tangents(mesh) for mesh in model.meshes])


def _apply_unconstrained_visibility_fallback(
    mesh: Mesh,
    sample_triangles: npt.NDArray[np.uint32],
    vertex_cones: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if sample_triangles.shape[0] == 0:
        vertex_cones[:, :3] = _normalize_rows(mesh.normals)
        vertex_cones[:, 3] = HALF_PI
        vertex_cones[:, 4] = 1.0
        return vertex_cones

    constrained = np.zeros((mesh.positions.shape[0],), dtype=np.bool_)
    valid_triangles = sample_triangles.astype(np.int64)
    valid_triangles = valid_triangles[(valid_triangles >= 0) & (valid_triangles < mesh.indices.shape[0])]
    if valid_triangles.shape[0] > 0:
        constrained[np.unique(mesh.indices[valid_triangles].reshape(-1).astype(np.int64))] = True
    unconstrained = ~constrained
    if np.any(unconstrained):
        vertex_cones[unconstrained, :3] = _normalize_rows(mesh.normals[unconstrained])
        vertex_cones[unconstrained, 3] = HALF_PI
        vertex_cones[unconstrained, 4] = 1.0
    return vertex_cones


def _scene_triangles(model: Model) -> npt.NDArray[np.float32]:
    parts = []
    for mesh in model.meshes:
        if mesh.indices.shape[0] > 0:
            parts.append(mesh.positions[mesh.indices.astype(np.int64)])
    if not parts:
        return np.zeros((0, 3, 3), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _intersects_any_bruteforce(
    origins: npt.NDArray[np.float32],
    directions: npt.NDArray[np.float32],
    triangles: npt.NDArray[np.float32],
    *,
    max_distance: float,
    ray_epsilon: float,
    ray_chunk_size: int = 256,
    triangle_chunk_size: int = 4096,
) -> npt.NDArray[np.bool_]:
    origins = np.asarray(origins, dtype=np.float32)
    directions = _normalize_rows(np.asarray(directions, dtype=np.float32))
    triangles = np.asarray(triangles, dtype=np.float32)
    hit_any = np.zeros((origins.shape[0],), dtype=np.bool_)
    if origins.shape[0] == 0 or triangles.shape[0] == 0:
        return hit_any

    max_distance = float(max_distance)
    ray_epsilon = float(ray_epsilon)
    for ray_begin in range(0, origins.shape[0], max(1, int(ray_chunk_size))):
        ray_end = min(origins.shape[0], ray_begin + max(1, int(ray_chunk_size)))
        ray_origins = origins[ray_begin:ray_end]
        ray_dirs = directions[ray_begin:ray_end]
        local_hit = np.zeros((ray_end - ray_begin,), dtype=np.bool_)
        for tri_begin in range(0, triangles.shape[0], max(1, int(triangle_chunk_size))):
            if np.all(local_hit):
                break
            tri_end = min(triangles.shape[0], tri_begin + max(1, int(triangle_chunk_size)))
            tri = triangles[tri_begin:tri_end].astype(np.float64)
            v0 = tri[:, 0]
            edge1 = tri[:, 1] - tri[:, 0]
            edge2 = tri[:, 2] - tri[:, 0]

            h = np.cross(ray_dirs[:, None, :].astype(np.float64), edge2[None, :, :])
            a = np.sum(edge1[None, :, :] * h, axis=2)
            valid = np.abs(a) > 1e-10
            f = np.zeros_like(a)
            f[valid] = 1.0 / a[valid]

            s = ray_origins[:, None, :].astype(np.float64) - v0[None, :, :]
            u = f * np.sum(s * h, axis=2)
            valid &= (u >= -1e-8) & (u <= 1.0 + 1e-8)

            q = np.cross(s, edge1[None, :, :])
            v = f * np.sum(ray_dirs[:, None, :].astype(np.float64) * q, axis=2)
            valid &= (v >= -1e-8) & (u + v <= 1.0 + 1e-8)

            t = f * np.sum(edge2[None, :, :] * q, axis=2)
            valid &= t > ray_epsilon
            if np.isfinite(max_distance):
                valid &= t < max_distance
            local_hit |= np.any(valid, axis=1)
        hit_any[ray_begin:ray_end] = local_hit
    return hit_any


def sample_visibility_cones_python(
    model: Model,
    samples: SurfaceSamples,
    *,
    ray_count: int = 128,
    max_distance: float = np.inf,
    self_bias: float | None = None,
) -> VisibilitySampleCones:
    mesh_tangents = [compute_mesh_tangents(mesh) for mesh in model.meshes]
    normals, tangents = _sample_frames(model, samples, mesh_tangents)
    local_dirs = _local_fibonacci_hemisphere(ray_count)
    sample_count = int(samples.positions.shape[0])
    if sample_count == 0:
        return VisibilitySampleCones(
            directions=np.zeros((0, 3), dtype=np.float32),
            aperture_radians=np.zeros((0,), dtype=np.float32),
            scale=np.zeros((0,), dtype=np.float32),
            visible_fraction=np.zeros((0,), dtype=np.float32),
        )

    repeated_normals = np.repeat(normals, ray_count, axis=0)
    repeated_tangents = np.repeat(tangents, ray_count, axis=0)
    tiled_local_dirs = np.tile(local_dirs, (sample_count, 1))
    ray_dirs = tangent_space_to_world(tiled_local_dirs, repeated_normals, repeated_tangents)

    scene_scale = max(float(np.linalg.norm(model.bounds_max - model.bounds_min)), 1.0)
    if self_bias is None:
        self_bias = scene_scale * 1e-5
    ray_origins = np.repeat(samples.positions, ray_count, axis=0) + repeated_normals * float(self_bias)
    hits = _intersects_any_bruteforce(
        ray_origins,
        ray_dirs,
        _scene_triangles(model),
        max_distance=float(max_distance),
        ray_epsilon=max(float(self_bias) * 0.5, 1e-7),
    ).reshape(sample_count, ray_count)

    visible = ~hits
    visible_fraction = np.mean(visible, axis=1).astype(np.float32)
    directions = np.zeros((sample_count, 3), dtype=np.float32)
    for sample_index in range(sample_count):
        if not np.any(visible[sample_index]):
            directions[sample_index] = normals[sample_index]
        else:
            begin = sample_index * ray_count
            end = begin + ray_count
            directions[sample_index] = np.sum(ray_dirs[begin:end][visible[sample_index]], axis=0)
    directions = _normalize_rows(directions, normals)
    aperture_radians = np.arccos(np.clip(1.0 - visible_fraction.astype(np.float64), 0.0, 1.0)).astype(np.float32)
    aperture_normalized = np.clip(aperture_radians / HALF_PI, 0.0, 1.0)
    scale = np.divide(
        visible_fraction,
        np.maximum(aperture_normalized, 1e-6),
        out=np.zeros_like(visible_fraction),
        where=aperture_normalized > 1e-6,
    )
    scale = np.clip(scale, 0.0, 1.0).astype(np.float32)
    return VisibilitySampleCones(
        directions=directions,
        aperture_radians=aperture_radians,
        scale=scale,
        visible_fraction=visible_fraction,
    )


def fit_visibility_cones(
    model: Model,
    samples: SurfaceSamples,
    sample_cones: VisibilitySampleCones,
    *,
    fit_backend: str = "python",
    regularization_weight: float = 0.0,
    build_native_first: bool = False,
) -> VisibilityBakeResult:
    sample_values = np.concatenate(
        [
            sample_cones.directions.astype(np.float32),
            sample_cones.aperture_radians[:, None].astype(np.float32),
            sample_cones.scale[:, None].astype(np.float32),
        ],
        axis=1,
    )
    if sample_values.shape[0] != samples.positions.shape[0]:
        raise ValueError("sample_cones length must match samples")

    backend = fit_backend.lower().replace("-", "_")
    if backend not in ("python", "native", "cpp", "cxx"):
        raise ValueError("fit_backend must be one of: python, native, cpp, cxx")
    use_native = backend in ("native", "cpp", "cxx")
    native_bake_visibility = None
    if use_native:
        from vertex_baking_utils import bake_visibility_least_squares, build_native

        if build_native_first:
            build_native(force=True)
        native_bake_visibility = bake_visibility_least_squares
    elif float(regularization_weight) != 0.0:
        raise ValueError("regularization_weight is only supported by the native visibility fit backend")

    mesh_tangents = [compute_mesh_tangents(mesh) for mesh in model.meshes]

    vertex_cones: list[npt.NDArray[np.float32]] = []
    encoded_texcoord2: list[npt.NDArray[np.float32]] = []
    for mesh_index, mesh in enumerate(model.meshes):
        mask = samples.mesh_indices == mesh_index
        if not np.any(mask):
            fallback = np.concatenate(
                [
                    _normalize_rows(mesh.normals).astype(np.float32),
                    np.full((mesh.positions.shape[0], 1), HALF_PI, dtype=np.float32),
                    np.ones((mesh.positions.shape[0], 1), dtype=np.float32),
                ],
                axis=1,
            )
            vertex_cones.append(fallback)
            encoded_texcoord2.append(
                encode_visibility_cone_texcoord2(
                    fallback[:, :3],
                    fallback[:, 3],
                    fallback[:, 4],
                    mesh.normals,
                    mesh_tangents[mesh_index],
                )
            )
            continue

        mesh_sample_triangles = samples.triangle_indices[mask].astype(np.uint32)
        mesh_sample_barycentrics = samples.barycentrics[mask]
        mesh_sample_values = sample_values[mask]
        if use_native:
            assert native_bake_visibility is not None
            solved, encoded = native_bake_visibility(
                mesh.positions,
                mesh.normals,
                mesh_tangents[mesh_index],
                mesh.indices,
                mesh_sample_triangles,
                mesh_sample_barycentrics,
                mesh_sample_values,
                regularization_weight=max(0.0, float(regularization_weight)),
            )
            solved = _apply_unconstrained_visibility_fallback(mesh, mesh_sample_triangles, solved)
            vertex_cones.append(solved.astype(np.float32, copy=False))
            encoded_texcoord2.append(encoded.astype(np.float32, copy=False))
        else:
            solved = bake_least_squares_python(
                mesh.positions,
                mesh.indices,
                mesh_sample_triangles,
                mesh_sample_barycentrics,
                mesh_sample_values,
            )
            solved[:, :3] = _normalize_rows(solved[:, :3], mesh.normals)
            solved[:, 3] = np.clip(solved[:, 3], 0.0, HALF_PI)
            solved[:, 4] = np.clip(solved[:, 4], 0.0, 1.0)
            solved = _apply_unconstrained_visibility_fallback(mesh, mesh_sample_triangles, solved)
            vertex_cones.append(solved.astype(np.float32))
            encoded_texcoord2.append(
                encode_visibility_cone_texcoord2(
                    vertex_cones[-1][:, :3],
                    vertex_cones[-1][:, 3],
                    vertex_cones[-1][:, 4],
                    mesh.normals,
                    mesh_tangents[mesh_index],
                )
            )

    return VisibilityBakeResult(
        samples=samples,
        sample_cones=sample_cones,
        vertex_cones=vertex_cones,
        encoded_texcoord2=encoded_texcoord2,
    )


def bake_visibility_cones_python(
    model: Model,
    samples: SurfaceSamples | None = None,
    *,
    sample_count: int = 4096,
    surface_seed: int = 1,
    min_samples_per_mesh: int = 0,
    visibility_ray_count: int = 128,
    max_distance: float = np.inf,
    self_bias: float | None = None,
    fit_backend: str = "python",
    regularization_weight: float = 0.0,
    build_native_first: bool = False,
) -> VisibilityBakeResult:
    if samples is None:
        samples = sample_model_surface(
            model,
            sample_count,
            seed=surface_seed,
            min_samples_per_mesh=max(0, int(min_samples_per_mesh)),
        )
    sample_cones = sample_visibility_cones_python(
        model,
        samples,
        ray_count=visibility_ray_count,
        max_distance=max_distance,
        self_bias=self_bias,
    )
    return fit_visibility_cones(
        model,
        samples,
        sample_cones,
        fit_backend=fit_backend,
        regularization_weight=regularization_weight,
        build_native_first=build_native_first,
    )


def fit_visibility_sh_pmr(
    model: Model,
    sampling,
    sample_sh,
    *,
    edge_regularization: float = 0.05,
    fit_backend: str = "python",
    build_native_first: bool = False,
) -> VisibilityBakeResult:
    from pmr_visibility_reference import (
        fit_pmr_vertex_sh,
        map_pmr_proxy_cones_to_model,
        pmr_vertex_sh_to_cones,
    )

    sample_sh = np.ascontiguousarray(sample_sh, dtype=np.float32)
    proxy_triangle_count = int(sampling.proxy.triangles.shape[0])
    samples_per_triangle = int(sampling.samples_per_triangle)
    expected_sample_count = proxy_triangle_count * samples_per_triangle
    if sampling.samples.positions.shape[0] != expected_sample_count:
        raise ValueError(
            "PMR sampling invariant failed: total samples must equal "
            "proxy_triangle_count * samples_per_triangle"
        )
    if sample_sh.shape != (expected_sample_count, 16):
        raise ValueError(f"sample_sh must have shape ({expected_sample_count}, 16)")
    backend = fit_backend.lower().replace("-", "_")
    if backend == "python":
        proxy_vertex_sh = fit_pmr_vertex_sh(
            sampling,
            sample_sh,
            edge_regularization=max(0.0, float(edge_regularization)),
        )
        proxy_cones = pmr_vertex_sh_to_cones(proxy_vertex_sh, sampling.proxy.normals)
    elif backend in ("native", "cpp", "cxx"):
        from vertex_baking_utils import (
            bake_pmr_visibility_sh_least_squares,
            build_native,
            pmr_sh_to_cones_native,
        )

        if build_native_first:
            build_native(force=True)
        proxy_vertex_sh = bake_pmr_visibility_sh_least_squares(
            sampling.proxy.positions,
            sampling.proxy.triangles,
            sampling.proxy.triangle_areas,
            sampling.samples_per_triangle,
            sampling.samples.barycentrics,
            sample_sh,
            edge_regularization=max(0.0, float(edge_regularization)),
        )
        proxy_cones = pmr_sh_to_cones_native(proxy_vertex_sh, sampling.proxy.normals)
    else:
        raise ValueError("fit_backend must be one of: python, native, cpp, cxx")
    vertex_cones, encoded_texcoord2 = map_pmr_proxy_cones_to_model(model, sampling.proxy, proxy_cones)
    return VisibilityBakeResult(
        samples=sampling.samples,
        sample_cones=None,
        vertex_cones=vertex_cones,
        encoded_texcoord2=encoded_texcoord2,
        sample_sh=sample_sh,
        proxy_vertex_sh=proxy_vertex_sh,
        sample_normals=sampling.sample_normals,
        samples_per_triangle=samples_per_triangle,
        proxy_triangle_count=proxy_triangle_count,
    )


def bake_visibility_cones_pmr_python(
    model: Model,
    *,
    samples_per_triangle: int = 256,
    visibility_ray_count: int = 512,
    ray_length: float = 0.5,
    self_bias: float = 0.001,
    edge_regularization: float = 0.05,
    proxy_voxel_size_mm: float = 0.1,
    proxy_compare_normals: bool = True,
    point_batch_size: int = 128,
    fit_backend: str = "python",
    build_native_first: bool = False,
) -> VisibilityBakeResult:
    """Run the published PMR visibility-tool pipeline on the CPU."""
    from pmr_visibility_reference import sample_model_surface_pmr, trace_pmr_visibility_sh_python

    sampling = sample_model_surface_pmr(
        model,
        samples_per_triangle=max(1, int(samples_per_triangle)),
        voxel_size_mm=float(proxy_voxel_size_mm),
        compare_normals=bool(proxy_compare_normals),
    )
    sample_sh = trace_pmr_visibility_sh_python(
        model,
        sampling,
        ray_count=max(1, int(visibility_ray_count)),
        ray_length=float(ray_length),
        self_bias=max(0.0, float(self_bias)),
        point_batch_size=max(1, int(point_batch_size)),
    )
    return fit_visibility_sh_pmr(
        model,
        sampling,
        sample_sh,
        edge_regularization=max(0.0, float(edge_regularization)),
        fit_backend=fit_backend,
        build_native_first=build_native_first,
    )


def save_visibility_npz(result: VisibilityBakeResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, npt.NDArray] = {
        "sample_positions": result.samples.positions,
        "sample_mesh_indices": result.samples.mesh_indices,
        "sample_triangle_indices": result.samples.triangle_indices,
        "sample_barycentrics": result.samples.barycentrics,
    }
    if result.sample_cones is not None:
        arrays.update(
            {
                "sample_cone_directions": result.sample_cones.directions,
                "sample_cone_aperture_radians": result.sample_cones.aperture_radians,
                "sample_cone_scale": result.sample_cones.scale,
                "sample_visible_fraction": result.sample_cones.visible_fraction,
            }
        )
    if result.sample_sh is not None:
        arrays["sample_visibility_sh16"] = result.sample_sh
        arrays["sample_visible_fraction"] = np.clip(
            result.sample_sh[:, 0] / np.sqrt(4.0 * np.pi),
            0.0,
            1.0,
        ).astype(np.float32)
    if result.proxy_vertex_sh is not None:
        arrays["proxy_vertex_visibility_sh16"] = result.proxy_vertex_sh
    if result.sample_normals is not None:
        arrays["sample_normals"] = result.sample_normals
    if result.samples_per_triangle is not None:
        arrays["samples_per_triangle"] = np.asarray(result.samples_per_triangle, dtype=np.uint32)
    if result.proxy_triangle_count is not None:
        arrays["proxy_triangle_count"] = np.asarray(result.proxy_triangle_count, dtype=np.uint32)
    for mesh_index, (raw, encoded) in enumerate(zip(result.vertex_cones, result.encoded_texcoord2)):
        arrays[f"mesh_{mesh_index}_raw_visibility_cone"] = raw
        arrays[f"mesh_{mesh_index}_texcoord2"] = encoded
    np.savez_compressed(output_path, **arrays)


def vertex_visibility_preview_values(
    result: VisibilityBakeResult,
    model: Model | None = None,
) -> list[npt.NDArray[np.float32]]:
    values = []
    for mesh_index, encoded in enumerate(result.encoded_texcoord2):
        aperture_normalized = np.clip(encoded[:, 2] / HALF_PI, 0.0, 1.0)
        if model is None:
            ambient_visibility = aperture_normalized * encoded[:, 3]
        else:
            cones = result.vertex_cones[mesh_index]
            normals = _normalize_rows(model.meshes[mesh_index].normals)
            cos_theta = np.clip(np.sum(_normalize_rows(cones[:, :3], normals) * normals, axis=1), 0.0, 1.0)
            corrected_cos_theta = (
                cos_theta * (1.0 - aperture_normalized)
                + (cos_theta * 0.5 + 0.5) * aperture_normalized
            )
            ambient_visibility = corrected_cos_theta * aperture_normalized * aperture_normalized * cones[:, 4]
        ambient_visibility = np.clip(ambient_visibility, 0.0, 1.0)
        values.append(ambient_visibility[:, None].astype(np.float32))
    return values
