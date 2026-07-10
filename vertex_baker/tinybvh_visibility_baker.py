from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from pmr_visibility_reference import (
    PMR_DEFAULT_RAY_COUNT,
    PMR_DEFAULT_RAY_LENGTH,
    PMR_DEFAULT_SELF_BIAS,
    sample_model_surface_pmr,
)
from vertex_baking_utils import build_native, trace_pmr_visibility_sh_tinybvh
from visibility_baker import fit_visibility_sh_pmr


def flatten_model_geometry(model) -> tuple[np.ndarray, np.ndarray]:
    position_parts = []
    index_parts = []
    vertex_offset = 0
    for mesh in model.meshes:
        positions = np.ascontiguousarray(mesh.positions, dtype=np.float32)
        indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
        if positions.shape[0] == 0 or indices.shape[0] == 0:
            continue
        position_parts.append(positions)
        index_parts.append(indices + np.uint32(vertex_offset))
        vertex_offset += int(positions.shape[0])
    if not position_parts:
        raise ValueError("model contains no triangles for TinyBVH visibility")
    return (
        np.ascontiguousarray(np.concatenate(position_parts, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(index_parts, axis=0), dtype=np.uint32),
    )


def bake_pmr_visibility_cones_tinybvh(
    model,
    *,
    samples_per_triangle: int = 256,
    visibility_ray_count: int = PMR_DEFAULT_RAY_COUNT,
    ray_length: float = PMR_DEFAULT_RAY_LENGTH,
    self_bias: float = PMR_DEFAULT_SELF_BIAS,
    edge_regularization: float = 0.05,
    proxy_voxel_size_mm: float = 0.1,
    proxy_compare_normals: bool = True,
    fit_backend: str = "native",
    build_native_first: bool = False,
    thread_count: int = 0,
    layout: str = "auto",
):
    total_begin = perf_counter()
    sampling_begin = perf_counter()
    sampling = sample_model_surface_pmr(
        model,
        samples_per_triangle=max(1, int(samples_per_triangle)),
        voxel_size_mm=float(proxy_voxel_size_mm),
        compare_normals=bool(proxy_compare_normals),
    )
    sampling_milliseconds = (perf_counter() - sampling_begin) * 1000.0
    positions, indices = flatten_model_geometry(model)

    if build_native_first:
        build_native(force=True)
    sample_sh, native_stats = trace_pmr_visibility_sh_tinybvh(
        positions,
        indices,
        sampling.samples.positions,
        sampling.sample_normals,
        ray_count=max(1, int(visibility_ray_count)),
        max_distance=float(ray_length),
        self_bias=max(0.0, float(self_bias)),
        thread_count=max(0, int(thread_count)),
        layout=layout,
    )

    fit_begin = perf_counter()
    result = fit_visibility_sh_pmr(
        model,
        sampling,
        sample_sh,
        edge_regularization=max(0.0, float(edge_regularization)),
        fit_backend=fit_backend,
        build_native_first=False,
    )
    fit_milliseconds = (perf_counter() - fit_begin) * 1000.0
    statistics = {
        "layout": native_stats.layout,
        "thread_count": native_stats.thread_count,
        "build_milliseconds": native_stats.build_milliseconds,
        "trace_milliseconds": native_stats.trace_milliseconds,
        "sampling_milliseconds": sampling_milliseconds,
        "fit_milliseconds": fit_milliseconds,
        "total_milliseconds": (perf_counter() - total_begin) * 1000.0,
        "ray_count": native_stats.total_ray_count,
        "visible_ray_count": native_stats.visible_ray_count,
        "visible_fraction": native_stats.visible_fraction,
        "million_rays_per_second": native_stats.rays_per_second * 1e-6,
    }
    return replace(result, trace_backend="tinybvh", trace_statistics=statistics)
