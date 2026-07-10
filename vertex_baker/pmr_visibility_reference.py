from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from model import Model
from surface_sampler import SurfaceSamples


PMR_SH_ORDER = 4
PMR_SH_COEFFICIENT_COUNT = 16
PMR_DEFAULT_SAMPLES_PER_TRIANGLE = 256
# Compatibility name for callers created before samples-per-triangle became the public budget.
PMR_DEFAULT_TRIANGLE_SAMPLE_COUNT = PMR_DEFAULT_SAMPLES_PER_TRIANGLE
PMR_DEFAULT_RAY_COUNT = 512
PMR_DEFAULT_RAY_LENGTH = 0.5
PMR_DEFAULT_SELF_BIAS = 0.001
PMR_DEFAULT_EDGE_REGULARIZATION = 0.05
PMR_DEFAULT_PROXY_VOXEL_SIZE_MM = 0.1


@dataclass(frozen=True)
class PMRMeshProxy:
    positions: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    mesh_vertex_remapping: tuple[npt.NDArray[np.uint32], ...]
    triangles: npt.NDArray[np.uint32]
    triangle_areas: npt.NDArray[np.float64]
    triangle_mesh_indices: npt.NDArray[np.int32]
    triangle_local_indices: npt.NDArray[np.int32]


@dataclass(frozen=True)
class PMRSurfaceSampling:
    proxy: PMRMeshProxy
    samples: SurfaceSamples
    sample_normals: npt.NDArray[np.float32]
    sample_proxy_triangle_indices: npt.NDArray[np.int32]
    samples_per_triangle: int


def _normalize_rows(values, fallback=None, eps: float = 1e-12) -> npt.NDArray[np.float32]:
    values = np.asarray(values, dtype=np.float64)
    result = values.copy()
    lengths = np.linalg.norm(result, axis=1)
    valid = lengths > eps
    result[valid] /= lengths[valid, None]
    if np.any(~valid):
        if fallback is None:
            result[~valid] = 0.0
        else:
            fb = np.asarray(fallback, dtype=np.float64)
            if fb.ndim == 1:
                result[~valid] = fb
            else:
                result[~valid] = fb[~valid]
            fb_lengths = np.linalg.norm(result[~valid], axis=1)
            fb_valid = fb_lengths > eps
            invalid_indices = np.nonzero(~valid)[0]
            result[invalid_indices[fb_valid]] /= fb_lengths[fb_valid, None]
            result[invalid_indices[~fb_valid]] = np.array([0.0, 0.0, 1.0])
    return result.astype(np.float32)


def build_pmr_mesh_proxy(
    model: Model,
    *,
    voxel_size_mm: float = PMR_DEFAULT_PROXY_VOXEL_SIZE_MM,
    compare_normals: bool = True,
    minimum_triangle_area: float = 1e-4,
) -> PMRMeshProxy:
    """Mirror MeshProxy.Build from the original Unity visibility tool."""
    voxel_size_mm = float(voxel_size_mm)
    if voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be positive")

    total_vertex_count = sum(int(mesh.positions.shape[0]) for mesh in model.meshes)
    proxy_positions = np.zeros((total_vertex_count, 3), dtype=np.float32)
    proxy_normals = np.zeros((total_vertex_count, 3), dtype=np.float32)
    mesh_remaps: list[npt.NDArray[np.uint32]] = []
    vertex_hash: dict[tuple[int, ...], int] = {}
    proxy_vertex_count = 0
    inv_voxel_size_m = 1000.0 / voxel_size_mm

    for mesh in model.meshes:
        positions = np.asarray(mesh.positions, dtype=np.float32)
        raw_normals = np.asarray(mesh.normals, dtype=np.float32)
        normals = _normalize_rows(raw_normals)
        hash_positions_value = getattr(mesh, "proxy_hash_positions", None)
        hash_normals_value = getattr(mesh, "proxy_hash_normals", None)
        hash_positions = positions if hash_positions_value is None else np.asarray(hash_positions_value, dtype=np.float32)
        hash_normals = raw_normals if hash_normals_value is None else np.asarray(hash_normals_value, dtype=np.float32)
        if hash_positions.shape != positions.shape or hash_normals.shape != raw_normals.shape:
            raise ValueError("proxy hash position/normal arrays must match the mesh vertex count")
        remap = np.zeros((positions.shape[0],), dtype=np.uint32)
        for vertex_index in range(positions.shape[0]):
            position_components = np.rint(
                (hash_positions[vertex_index].astype(np.float64) + 50.0) * inv_voxel_size_m
            ).astype(np.int64)
            normal_source = hash_normals[vertex_index] if compare_normals else np.zeros((3,), dtype=np.float32)
            normal_components = np.rint(
                (normal_source.astype(np.float64) + 50.0) * 10000.0
            ).astype(np.int64)
            position_key = (
                int(position_components[0]) * 1_000_000_000_000
                + int(position_components[1]) * 1_000_000
                + int(position_components[2])
            )
            normal_key = (
                int(normal_components[0]) * 1_000_000_000_000
                + int(normal_components[1]) * 1_000_000
                + int(normal_components[2])
            )
            key = (position_key, normal_key)
            proxy_index = vertex_hash.get(key)
            if proxy_index is None:
                proxy_index = proxy_vertex_count
                proxy_vertex_count += 1
                vertex_hash[key] = proxy_index
            remap[vertex_index] = proxy_index
            # Unity overwrites the representative with the latest matching vertex.
            proxy_positions[proxy_index] = positions[vertex_index]
            proxy_normals[proxy_index] = normals[vertex_index]
        mesh_remaps.append(remap)

    proxy_positions = proxy_positions[:proxy_vertex_count]
    proxy_normals = proxy_normals[:proxy_vertex_count]

    proxy_triangles: list[npt.NDArray[np.uint32]] = []
    triangle_mesh_indices: list[npt.NDArray[np.int32]] = []
    triangle_local_indices: list[npt.NDArray[np.int32]] = []
    for mesh_index, mesh in enumerate(model.meshes):
        indices = np.asarray(mesh.indices, dtype=np.uint32)
        remapped = mesh_remaps[mesh_index][indices.astype(np.int64)]
        proxy_triangles.append(remapped.astype(np.uint32, copy=False))
        triangle_count = int(indices.shape[0])
        triangle_mesh_indices.append(np.full((triangle_count,), mesh_index, dtype=np.int32))
        triangle_local_indices.append(np.arange(triangle_count, dtype=np.int32))

    triangles = np.concatenate(proxy_triangles, axis=0) if proxy_triangles else np.zeros((0, 3), dtype=np.uint32)
    tri_positions = proxy_positions[triangles.astype(np.int64)]
    raw_areas = 0.5 * np.linalg.norm(
        np.cross(tri_positions[:, 1] - tri_positions[:, 0], tri_positions[:, 2] - tri_positions[:, 0]),
        axis=1,
    )
    # The shipped C# clamps before its zero-area test, so every triangle remains.
    triangle_areas = np.maximum(raw_areas.astype(np.float64), float(minimum_triangle_area))

    return PMRMeshProxy(
        positions=proxy_positions,
        normals=proxy_normals,
        mesh_vertex_remapping=tuple(mesh_remaps),
        triangles=triangles,
        triangle_areas=triangle_areas,
        triangle_mesh_indices=np.concatenate(triangle_mesh_indices) if triangle_mesh_indices else np.zeros((0,), dtype=np.int32),
        triangle_local_indices=np.concatenate(triangle_local_indices) if triangle_local_indices else np.zeros((0,), dtype=np.int32),
    )


def _reverse_bits_u32(value: int) -> int:
    value &= 0xFFFFFFFF
    value = ((value << 16) | (value >> 16)) & 0xFFFFFFFF
    value = (((value & 0x55555555) << 1) | ((value & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
    value = (((value & 0x33333333) << 2) | ((value & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
    value = (((value & 0x0F0F0F0F) << 4) | ((value & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF
    value = (((value & 0x00FF00FF) << 8) | ((value & 0xFF00FF00) >> 8)) & 0xFFFFFFFF
    return value


def pmr_triangle_barycentrics(samples_per_triangle: int) -> npt.NDArray[np.float32]:
    """Basu-Owen triangle sequence used by TriangleSamples in the C# tool."""
    samples_per_triangle = int(samples_per_triangle)
    if samples_per_triangle < 1:
        raise ValueError("samples_per_triangle must be positive")

    result = np.zeros((samples_per_triangle, 3), dtype=np.float32)
    for sample_index in range(samples_per_triangle):
        fixed_point = _reverse_bits_u32(sample_index)
        a = np.array([1.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0], dtype=np.float64)
        c = np.array([0.0, 0.0], dtype=np.float64)
        for digit_index in range(16):
            digit = (fixed_point >> (2 * (15 - digit_index))) & 0x3
            if digit == 0:
                an, bn, cn = (b + c) * 0.5, (a + c) * 0.5, (a + b) * 0.5
            elif digit == 1:
                an, bn, cn = a, (a + b) * 0.5, (a + c) * 0.5
            elif digit == 2:
                an, bn, cn = (b + a) * 0.5, b, (b + c) * 0.5
            else:
                an, bn, cn = (c + a) * 0.5, (c + b) * 0.5, c
            a, b, c = an, bn, cn
        uv = (a + b + c) / 3.0
        result[sample_index] = np.array([uv[0], uv[1], 1.0 - uv[0] - uv[1]], dtype=np.float32)
    return result


def sample_model_surface_pmr(
    model: Model,
    *,
    samples_per_triangle: int = PMR_DEFAULT_SAMPLES_PER_TRIANGLE,
    voxel_size_mm: float = PMR_DEFAULT_PROXY_VOXEL_SIZE_MM,
    compare_normals: bool = True,
) -> PMRSurfaceSampling:
    proxy = build_pmr_mesh_proxy(model, voxel_size_mm=voxel_size_mm, compare_normals=compare_normals)
    samples_per_triangle = int(samples_per_triangle)
    if samples_per_triangle < 1:
        raise ValueError("samples_per_triangle must be positive")
    barycentric_pattern = pmr_triangle_barycentrics(samples_per_triangle)
    triangle_count = int(proxy.triangles.shape[0])
    sample_proxy_triangles = np.repeat(np.arange(triangle_count, dtype=np.int32), samples_per_triangle)
    barycentrics = np.tile(barycentric_pattern, (triangle_count, 1))
    tri = proxy.positions[proxy.triangles.astype(np.int64)]
    sample_tri_positions = tri[sample_proxy_triangles]
    positions = np.sum(sample_tri_positions * barycentrics[:, :, None], axis=1).astype(np.float32)

    edge01 = tri[:, 1] - tri[:, 0]
    edge02 = tri[:, 2] - tri[:, 0]
    face_normals = _normalize_rows(np.cross(_normalize_rows(edge01), edge02))
    sample_normals = np.repeat(face_normals, samples_per_triangle, axis=0)
    samples = SurfaceSamples(
        positions=positions,
        barycentrics=barycentrics,
        mesh_indices=np.repeat(proxy.triangle_mesh_indices, samples_per_triangle),
        triangle_indices=np.repeat(proxy.triangle_local_indices, samples_per_triangle),
    )
    return PMRSurfaceSampling(
        proxy=proxy,
        samples=samples,
        sample_normals=sample_normals,
        sample_proxy_triangle_indices=sample_proxy_triangles,
        samples_per_triangle=samples_per_triangle,
    )


def make_pmr_ray_directions(ray_count: int) -> npt.NDArray[np.float32]:
    ray_count = int(ray_count)
    if ray_count < 1:
        raise ValueError("ray_count must be positive")
    result = np.zeros((ray_count, 3), dtype=np.float32)
    for ray_index in range(ray_count):
        u0 = np.float32(ray_index / ray_count)
        u1 = np.float32(_reverse_bits_u32(ray_index) * 2.3283064365386963e-10)
        phi = np.float32(u0 * np.float32(2.0 * np.pi))
        cos_theta = np.float32(1.0 - np.float32(2.0) * u1)
        sin_theta = np.float32(np.sqrt(max(1.0 - float(cos_theta * cos_theta), 0.0)))
        result[ray_index] = np.array(
            [np.cos(phi) * sin_theta, cos_theta, np.sin(phi) * sin_theta],
            dtype=np.float32,
        )
    return result


def pmr_sh_basis(directions) -> npt.NDArray[np.float64]:
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (N, 3)")
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    basis = np.empty((directions.shape[0], PMR_SH_COEFFICIENT_COUNT), dtype=np.float64)
    basis[:, 0] = 0.28209479177387814
    basis[:, 1] = -0.4886025119029199 * y
    basis[:, 2] = 0.4886025119029199 * z
    basis[:, 3] = -0.4886025119029199 * x
    basis[:, 4] = 1.0925484305920792 * x * y
    basis[:, 5] = -1.0925484305920792 * y * z
    basis[:, 6] = 0.31539156525252005 * (-1.0 + 3.0 * z * z)
    basis[:, 7] = -1.0925484305920792 * x * z
    basis[:, 8] = 0.5462742152960396 * (x * x - y * y)
    basis[:, 9] = -0.5900435899266435 * (3.0 * x * x * y - y * y * y)
    basis[:, 10] = 2.890611442640554 * x * y * z
    basis[:, 11] = -0.4570457994644658 * y * (-1.0 + 5.0 * z * z)
    basis[:, 12] = 0.3731763325901154 * z * (-3.0 + 5.0 * z * z)
    basis[:, 13] = -0.4570457994644658 * x * (-1.0 + 5.0 * z * z)
    basis[:, 14] = 1.445305721320277 * (x * x - y * y) * z
    basis[:, 15] = -0.5900435899266435 * (x * x * x - 3.0 * x * y * y)
    return basis


def project_pmr_visibility_sh(visibility, ray_directions) -> npt.NDArray[np.float32]:
    visibility = np.asarray(visibility, dtype=np.float64)
    ray_directions = np.asarray(ray_directions, dtype=np.float32)
    if visibility.ndim != 2 or visibility.shape[1] != ray_directions.shape[0]:
        raise ValueError("visibility must have shape (sample_count, ray_count)")
    weight = (4.0 * np.pi) / float(ray_directions.shape[0])
    return (visibility @ pmr_sh_basis(ray_directions) * weight).astype(np.float32)


def trace_pmr_visibility_sh_python(
    model: Model,
    sampling: PMRSurfaceSampling,
    *,
    ray_count: int = PMR_DEFAULT_RAY_COUNT,
    ray_length: float = PMR_DEFAULT_RAY_LENGTH,
    self_bias: float = PMR_DEFAULT_SELF_BIAS,
    point_batch_size: int = 128,
) -> npt.NDArray[np.float32]:
    from visibility_baker import _intersects_any_bruteforce, _scene_triangles

    ray_directions = make_pmr_ray_directions(ray_count)
    sh_basis = pmr_sh_basis(ray_directions)
    sh_weight = (4.0 * np.pi) / float(ray_count)
    sample_count = int(sampling.samples.positions.shape[0])
    result = np.zeros((sample_count, PMR_SH_COEFFICIENT_COUNT), dtype=np.float32)
    triangles = _scene_triangles(model)
    point_batch_size = max(1, int(point_batch_size))
    for begin in range(0, sample_count, point_batch_size):
        end = min(sample_count, begin + point_batch_size)
        origins = sampling.samples.positions[begin:end] + sampling.sample_normals[begin:end] * float(self_bias)
        ray_origins = np.repeat(origins, ray_count, axis=0)
        ray_dirs = np.tile(ray_directions, (end - begin, 1))
        hits = _intersects_any_bruteforce(
            ray_origins,
            ray_dirs,
            triangles,
            max_distance=float(ray_length),
            ray_epsilon=0.0,
        ).reshape(end - begin, ray_count)
        result[begin:end] = ((~hits).astype(np.float64) @ sh_basis * sh_weight).astype(np.float32)
    return result


def _pmr_gradient_basis(proxy: PMRMeshProxy) -> npt.NDArray[np.float64]:
    tri = proxy.positions[proxy.triangles.astype(np.int64)].astype(np.float64)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    face0 = np.cross(v1 - v0, v2 - v1)
    face1 = np.cross(v2 - v1, v0 - v2)
    face2 = np.cross(v0 - v2, v1 - v0)

    def normalize(v):
        lengths = np.linalg.norm(v, axis=1)
        out = np.zeros_like(v)
        valid = lengths > 1e-20
        out[valid] = v[valid] / lengths[valid, None]
        return out

    scale = (0.5 / np.sqrt(proxy.triangle_areas))[:, None]
    return np.stack(
        [
            np.cross(normalize(face0), v2 - v1) * scale,
            np.cross(normalize(face1), v0 - v2) * scale,
            np.cross(normalize(face2), v1 - v0) * scale,
        ],
        axis=1,
    )


def build_pmr_sparse_matrix(proxy: PMRMeshProxy, edge_regularization: float = PMR_DEFAULT_EDGE_REGULARIZATION):
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError("PMR reference fitting requires scipy") from exc

    triangles = proxy.triangles.astype(np.int64)
    areas = proxy.triangle_areas
    rows: list[npt.NDArray[np.int64]] = []
    cols: list[npt.NDArray[np.int64]] = []
    values: list[npt.NDArray[np.float64]] = []
    for i in range(3):
        for j in range(3):
            rows.append(triangles[:, i])
            cols.append(triangles[:, j])
            values.append(areas * (1.0 / 6.0 if i == j else 1.0 / 12.0))

    edge_regularization = max(0.0, float(edge_regularization))
    if edge_regularization > 0.0 and triangles.shape[0] > 0:
        gradient_basis = _pmr_gradient_basis(proxy)
        edge_links: dict[tuple[int, int], list[int]] = {}
        for triangle_index, ids in enumerate(triangles):
            for edge_index in range(3):
                va = int(ids[edge_index])
                vb = int(ids[(edge_index + 1) % 3])
                edge_links.setdefault((min(va, vb), max(va, vb)), []).append(triangle_index)

        reg_rows: list[int] = []
        reg_cols: list[int] = []
        reg_values: list[float] = []
        for linked_triangles in edge_links.values():
            for triangle_a, triangle_b in combinations(linked_triangles, 2):
                ids_a = [int(v) for v in triangles[triangle_a]]
                ids_union = ids_a + [ids_a[0]]
                redge = np.zeros((4, 3), dtype=np.float64)
                redge[:3] = gradient_basis[triangle_a]
                unmatched_count = 0
                for corner_b, vertex_b in enumerate(triangles[triangle_b]):
                    vertex_b = int(vertex_b)
                    try:
                        union_index = ids_a.index(vertex_b)
                    except ValueError:
                        union_index = 3
                        ids_union[3] = vertex_b
                        unmatched_count += 1
                    redge[union_index] -= gradient_basis[triangle_b, corner_b]
                if unmatched_count > 1:
                    continue
                area = areas[triangle_a] + areas[triangle_b]
                gram = redge @ redge.T * (edge_regularization * area)
                for i in range(4):
                    for j in range(4):
                        reg_rows.append(ids_union[i])
                        reg_cols.append(ids_union[j])
                        reg_values.append(float(gram[i, j]))
        if reg_rows:
            rows.append(np.asarray(reg_rows, dtype=np.int64))
            cols.append(np.asarray(reg_cols, dtype=np.int64))
            values.append(np.asarray(reg_values, dtype=np.float64))

    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(proxy.positions.shape[0], proxy.positions.shape[0]),
        dtype=np.float64,
    )
    return matrix


def fit_pmr_vertex_sh(
    sampling: PMRSurfaceSampling,
    sample_sh,
    *,
    edge_regularization: float = PMR_DEFAULT_EDGE_REGULARIZATION,
) -> npt.NDArray[np.float32]:
    try:
        from scipy.sparse.linalg import factorized
    except ImportError as exc:
        raise RuntimeError("PMR reference fitting requires scipy") from exc

    sample_sh = np.asarray(sample_sh, dtype=np.float32)
    sample_count = int(sampling.samples.positions.shape[0])
    if sample_sh.shape != (sample_count, PMR_SH_COEFFICIENT_COUNT):
        raise ValueError(f"sample_sh must have shape ({sample_count}, {PMR_SH_COEFFICIENT_COUNT})")

    proxy = sampling.proxy
    rhs = np.zeros((proxy.positions.shape[0], PMR_SH_COEFFICIENT_COUNT), dtype=np.float64)
    sample_triangle_ids = sampling.sample_proxy_triangle_indices.astype(np.int64)
    sample_vertex_ids = proxy.triangles[sample_triangle_ids].astype(np.int64)
    sample_weight = proxy.triangle_areas[sample_triangle_ids] / float(sampling.samples_per_triangle)
    weighted_sh = sample_sh.astype(np.float64) * sample_weight[:, None]
    for corner in range(3):
        contribution = weighted_sh * sampling.samples.barycentrics[:, corner : corner + 1].astype(np.float64)
        np.add.at(rhs, sample_vertex_ids[:, corner], contribution)

    matrix = build_pmr_sparse_matrix(proxy, edge_regularization=edge_regularization).tocsc()
    solve = factorized(matrix)
    result = np.zeros_like(rhs)
    for coefficient_index in range(PMR_SH_COEFFICIENT_COUNT):
        result[:, coefficient_index] = solve(rhs[:, coefficient_index])
    return result.astype(np.float32)


_PMR_ROTATION_LOBES = np.array(
    [
        [3.1416, 2.6180],
        [1.5708, -2.6180],
        [1.5708, 1.5708],
        [2.0344, -3.1416],
        [2.0344, -1.5708],
        [2.0344, -0.5236],
        [2.0344, 1.5708],
    ],
    dtype=np.float64,
)


def _pmr_lobe_directions() -> npt.NDArray[np.float64]:
    theta = _PMR_ROTATION_LOBES[:, 0]
    phi = _PMR_ROTATION_LOBES[:, 1]
    sin_theta = np.sin(theta)
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=1)


def _pmr_a_hat_transpose(sh: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    z = np.zeros((PMR_SH_COEFFICIENT_COUNT,), dtype=np.float64)
    z[4] = 1.58533 * sh[4] + 0.457646 * sh[5] + 1.58533 * sh[6] - 1.37294 * sh[7] - 0.915291 * sh[8]
    z[5] = 2.11378 * sh[4]
    z[6] = 1.05689 * sh[4] + 1.83058 * sh[5] - 1.83058 * sh[7] - 1.83058 * sh[8]
    z[7] = -2.28823 * sh[7]
    z[8] = -2.28823 * sh[5]
    z[9] = 1.498 * sh[10] - 1.33985 * sh[12] + 0.864869 * sh[14] + 2.11849 * sh[15]
    z[10] = -2.52644 * sh[13]
    z[11] = 2.18796 * sh[11] - 1.26322 * sh[13]
    z[12] = 2.36854 * sh[15]
    z[13] = -1.18427 * sh[9] + 1.67481 * sh[10] + 1.52889 * sh[11] - 2.64811 * sh[13] + 0.966953 * sh[14] + 1.18427 * sh[15]
    z[14] = 2.23308 * sh[10]
    z[15] = 1.18427 * sh[9] - 0.55827 * sh[10] - 1.52889 * sh[11] + 2.64811 * sh[13] + 0.966953 * sh[14] + 1.18427 * sh[15]
    return z


def _pmr_rotation_inverse(axis: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    new_y = np.asarray(axis, dtype=np.float64)
    axis_length = float(np.linalg.norm(new_y))
    if axis_length <= 1e-20:
        # Unity returns a zero inverse for the singular rotation built from a zero lobe axis.
        return np.zeros((3, 3), dtype=np.float64)
    new_y = new_y / axis_length
    if new_y[1] >= 0.999:
        new_x = np.array([1.0, 0.0, 0.0])
        new_z = np.array([0.0, 0.0, 1.0])
    elif new_y[1] <= -0.999:
        new_x = np.array([1.0, 0.0, 0.0])
        new_z = np.array([0.0, 0.0, -1.0])
    else:
        new_x = np.cross(np.array([0.0, 1.0, 0.0]), new_y)
        new_z = np.cross(new_x, new_y)
        new_x /= np.linalg.norm(new_x)
        new_y /= np.linalg.norm(new_y)
        new_z /= np.linalg.norm(new_z)
    rotation = np.column_stack([new_x, new_z, new_y])
    return np.linalg.inv(rotation)


def rotate_pmr_sh_to_zonal(sh, fallback_direction=None) -> npt.NDArray[np.float64]:
    sh = np.asarray(sh, dtype=np.float64)
    if sh.shape != (PMR_SH_COEFFICIENT_COUNT,):
        raise ValueError("sh must have shape (16,)")
    linear_direction = np.array([-sh[3], -sh[1], sh[2]], dtype=np.float64)
    linear_length = float(np.linalg.norm(linear_direction))
    if linear_length > 1e-20:
        axis = linear_direction / linear_length
    else:
        axis = np.zeros((3,), dtype=np.float64)

    rotation_inverse = _pmr_rotation_inverse(axis)
    rotated_lobes = (rotation_inverse @ _pmr_lobe_directions().T).T
    rotated_basis = pmr_sh_basis(rotated_lobes)
    z = _pmr_a_hat_transpose(sh)

    out = np.zeros((PMR_SH_COEFFICIENT_COUNT,), dtype=np.float64)
    out[0] = sh[0]
    out[2] = (rotation_inverse @ linear_direction)[2]
    out[6] = float(np.dot(rotated_basis[:5, 6], z[4:9]))
    out[12] = float(np.dot(rotated_basis[:7, 12], z[9:16]))
    return out


def _pmr_cone_coefficient(aperture: float, coefficient_index: int) -> float:
    if coefficient_index == 0:
        return -np.sqrt(np.pi) * (-1.0 + np.cos(aperture))
    if coefficient_index == 2:
        return 0.5 * np.sqrt(3.0 * np.pi) * np.sin(aperture) ** 2
    if coefficient_index == 6:
        return 0.5 * np.sqrt(5.0 * np.pi) * np.sin(aperture) ** 2 * np.cos(aperture)
    if coefficient_index == 12:
        return (np.sqrt(7.0 * np.pi) / 16.0) * np.sin(aperture) ** 2 * (5.0 * np.cos(2.0 * aperture) + 3.0)
    return 0.0


def _pmr_cone_coefficient_derivative(aperture: float, coefficient_index: int) -> float:
    if coefficient_index == 0:
        return np.sqrt(np.pi) * np.sin(aperture)
    if coefficient_index == 2:
        return np.sqrt(3.0 * np.pi) * np.cos(aperture) * np.sin(aperture)
    if coefficient_index == 6:
        return (3.0 * np.sqrt(5.0 * np.pi) * np.sin(3.0 * aperture) - np.sqrt(5.0 * np.pi) * np.sin(aperture)) / 8.0
    if coefficient_index == 12:
        return (5.0 * np.sqrt(7.0 * np.pi) * np.sin(4.0 * aperture) - 2.0 * np.sqrt(7.0 * np.pi) * np.sin(2.0 * aperture)) / 16.0
    return 0.0


def _pmr_aperture_equation(zonal_sh: npt.NDArray[np.float64], aperture: float) -> float:
    coefficients = np.array([_pmr_cone_coefficient(aperture, i) for i in range(PMR_SH_COEFFICIENT_COUNT)])
    derivatives = np.array([_pmr_cone_coefficient_derivative(aperture, i) for i in range(PMR_SH_COEFFICIENT_COUNT)])
    numerator = float(np.dot(zonal_sh, coefficients))
    denominator = float(np.dot(coefficients, coefficients))
    return float(np.dot(zonal_sh, derivatives) - np.dot(coefficients, derivatives) * numerator / denominator)


def solve_pmr_cone_angle_scale(zonal_sh) -> tuple[float, float]:
    zonal_sh = np.asarray(zonal_sh, dtype=np.float64)
    x1 = 0.001
    x2 = np.pi - 0.001
    split_count = 20
    step = (x2 - x1) / split_count
    values = np.array([_pmr_aperture_equation(zonal_sh, x1 + step * i) for i in range(split_count + 1)])
    found = False
    for i in range(split_count):
        for j in range(i + 1, split_count + 1):
            if values[i] * values[j] < 0.0:
                x2 = x1 + step * j
                x1 = x1 + step * i
                found = True
                break
        if found:
            break
    if not found:
        return (np.pi - 0.001 if zonal_sh[0] > 0.1 else 0.001), 1.0

    f = _pmr_aperture_equation(zonal_sh, x1)
    if f < 0.0:
        dx = x2 - x1
        aperture = x1
    else:
        dx = x1 - x2
        aperture = x2
    for _ in range(10):
        dx *= 0.5
        midpoint = aperture + dx
        midpoint_value = _pmr_aperture_equation(zonal_sh, midpoint)
        if midpoint_value <= 0.0:
            aperture = midpoint
        if midpoint_value == 0.0:
            break

    coefficients = np.array([_pmr_cone_coefficient(aperture, i) for i in range(PMR_SH_COEFFICIENT_COUNT)])
    scale = float(np.dot(zonal_sh, coefficients) / np.dot(coefficients, coefficients))
    return float(aperture), scale


def pmr_vertex_sh_to_cones(vertex_sh, fallback_normals=None) -> npt.NDArray[np.float32]:
    vertex_sh = np.asarray(vertex_sh, dtype=np.float32)
    if vertex_sh.ndim != 2 or vertex_sh.shape[1] != PMR_SH_COEFFICIENT_COUNT:
        raise ValueError("vertex_sh must have shape (N, 16)")
    linear = np.stack([-vertex_sh[:, 3], -vertex_sh[:, 1], vertex_sh[:, 2]], axis=1)
    directions = _normalize_rows(linear)
    cones = np.zeros((vertex_sh.shape[0], 5), dtype=np.float32)
    cones[:, :3] = directions
    for vertex_index in range(vertex_sh.shape[0]):
        zonal = rotate_pmr_sh_to_zonal(vertex_sh[vertex_index], fallback_direction=directions[vertex_index])
        aperture, scale = solve_pmr_cone_angle_scale(zonal)
        cones[vertex_index, 3] = aperture
        cones[vertex_index, 4] = scale
    return cones


def map_pmr_proxy_cones_to_model(
    model: Model,
    proxy: PMRMeshProxy,
    proxy_cones,
) -> tuple[list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]:
    from visibility_baker import compute_mesh_tangents, encode_visibility_cone_texcoord2

    proxy_cones = np.asarray(proxy_cones, dtype=np.float32)
    vertex_cones: list[npt.NDArray[np.float32]] = []
    encoded_texcoord2: list[npt.NDArray[np.float32]] = []
    for mesh_index, mesh in enumerate(model.meshes):
        cones = proxy_cones[proxy.mesh_vertex_remapping[mesh_index].astype(np.int64)]
        tangents = compute_mesh_tangents(mesh)
        encoded = encode_visibility_cone_texcoord2(
            cones[:, :3],
            cones[:, 3],
            cones[:, 4],
            mesh.normals,
            tangents,
            clamp_cone=False,
        )
        vertex_cones.append(cones.astype(np.float32, copy=False))
        encoded_texcoord2.append(encoded)
    return vertex_cones, encoded_texcoord2
